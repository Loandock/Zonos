import json
from typing import Callable, Generator

import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from zonos.autoencoder import DACAutoencoder
from zonos.backbone import BACKBONES
from zonos.codebook_pattern import apply_delay_pattern, revert_delay_pattern
from zonos.conditioning import PrefixConditioner
from zonos.config import InferenceParams, ZonosConfig
from zonos.sampling import sample_from_logits
from zonos.speaker_cloning import SpeakerEmbeddingLDA
from zonos.utils import DEFAULT_DEVICE, find_multiple, pad_weight_

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))

# H100 Optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        # TODO: pad to multiple of at least 8
        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        if config.pad_vocab_to_multiple_of:
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    def _pad_embeddings_and_heads(self, *args, **kwargs):
        for w in [*self.embeddings, *self.heads]:
            pad_weight_(w, self.config.pad_vocab_to_multiple_of)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, device: str = DEFAULT_DEVICE, **kwargs
    ) -> "Zonos":
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)
        return cls.from_local(config_path, model_path, device, **kwargs)

    @classmethod
    def from_local(
        cls, config_path: str, model_path: str, device: str = DEFAULT_DEVICE, backbone: str | None = None
    ) -> "Zonos":
        config = ZonosConfig.from_dict(json.load(open(config_path)))
        if backbone:
            backbone_cls = BACKBONES[backbone]
        else:
            is_transformer = not bool(config.backbone.ssm_cfg)
            backbone_cls = DEFAULT_BACKBONE_CLS
            # Preferentially route to pure torch backbone for increased performance and lower latency.
            if is_transformer and "torch" in BACKBONES:
                backbone_cls = BACKBONES["torch"]

        # H100 optimization: use bfloat16 for better performance
        model = cls(config, backbone_cls).to(device, torch.bfloat16)
        model.autoencoder.dac.to(device)

        sd = model.state_dict()
        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        model.load_state_dict(sd)

        return model

    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Generate a speaker embedding from an audio clip."""
        if self.spk_clone_model is None:
            self.spk_clone_model = SpeakerEmbeddingLDA()
        _, spk_embedding = self.spk_clone_model(wav.to(self.spk_clone_model.device), sr)
        return spk_embedding.unsqueeze(0).bfloat16()

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=1)

    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams, cfg_scale: float
    ) -> torch.Tensor:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        logits = self.apply_heads(last_hidden_states).squeeze(2).float()
        if cfg_scale != 1.0:
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        logits[..., 1025:].fill_(-torch.inf)  # ensures padding is ignored
        return logits

    def _decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True,
    ) -> torch.Tensor:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `_compute_logits`.

        Below we wrap this function with a simple CUDA Graph capturing mechanism,
        doing 3 warmup steps if needed and then capturing or replaying the graph.
        We only recapture if the batch size changes.
        """
        # H100 optimization: use CUDA graphs more aggressively
        if cfg_scale == 1.0:
            hidden_states = self.embed_codes(input_ids)
            return self._compute_logits(hidden_states, inference_params, cfg_scale)

        bsz = input_ids.size(0)

        if not allow_cudagraphs or input_ids.device.type != "cuda":
            hidden_states_local = self.embed_codes(input_ids)
            hidden_states_local = hidden_states_local.repeat(2, 1, 1)
            return self._compute_logits(hidden_states_local, inference_params, cfg_scale)

        need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz)

        if need_capture:
            self._cg_graph = None

            self._cg_batch_size = bsz
            self._cg_inference_params = inference_params
            self._cg_scale = cfg_scale

            # H100 optimization: more warmup iterations for better kernel selection
            for _ in range(5):  # Increased from 3 to 5
                hidden_states = self.embed_codes(input_ids)
                hidden_states = hidden_states.repeat(2, 1, 1)  # because cfg != 1.0
                logits = self._compute_logits(hidden_states, inference_params, cfg_scale)

            self._cg_input_ids = input_ids.clone()
            self._cg_logits = torch.empty_like(logits)

            g = torch.cuda.CUDAGraph()

            def capture_region():
                hidden_states_local = self.embed_codes(self._cg_input_ids)
                hidden_states_local = hidden_states_local.repeat(2, 1, 1)
                self._cg_logits = self._compute_logits(hidden_states_local, self._cg_inference_params, self._cg_scale)

            with torch.cuda.graph(g):
                capture_region()

            self._cg_graph = g

        else:
            self._cg_input_ids.copy_(input_ids)

        self._cg_graph.replay()

        return self._cg_logits

    def _prefill(
        self,
        prefix_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        # Replicate input_ids if CFG is enabled
        if cfg_scale != 1.0:
            input_ids = input_ids.expand(prefix_hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([prefix_hidden_states, self.embed_codes(input_ids)], dim=1)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16) -> InferenceParams:
        # H100 optimization: ensure sequence length is optimal for H100
        max_seqlen = find_multiple(max_seqlen, 16)  # Changed from 8 to 16 for H100
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32)
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> torch.Tensor:
        if uncond_dict is None:
            uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}
        return torch.cat(
            [
                self.prefix_conditioner(cond_dict),
                self.prefix_conditioner(uncond_dict),
            ]
        )

    def can_use_cudagraphs(self) -> bool:
        # H100 optimization: always try to use CUDA graphs on H100
        return self.device.type == "cuda"

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [bsz, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor | None = None,  # [bsz, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1,
        sampling_params: dict = dict(min_p=0.1),
        progress_bar: bool = True,
        disable_torch_compile: bool = False,
        callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    ):
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
        device = self.device

        # H100 optimization: use CUDA graphs more aggressively
        cg = self.can_use_cudagraphs()
        decode_one_token = self._decode_one_token
        
        # H100 optimization: use torch.compile with better settings
        if not cg and not disable_torch_compile:
            decode_one_token = torch.compile(
                decode_one_token, 
                dynamic=True, 
                mode="reduce-overhead",  # Better for H100
                fullgraph=True  # Better for H100
            )

        unknown_token = -1
        audio_seq_len = prefix_audio_len + max_new_tokens
        seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9

        with torch.device(device):
            # H100 optimization: ensure sequence length is optimal
            seq_len = find_multiple(seq_len, 16)
            inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
            codes = torch.full((batch_size, 9, audio_seq_len), unknown_token)

        if audio_prefix_codes is not None:
            codes[..., :prefix_audio_len] = audio_prefix_codes

        delayed_codes = apply_delay_pattern(codes, self.masked_token_id)

        delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

        logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
        next_token = sample_from_logits(logits, **sampling_params)

        offset = delayed_prefix_audio_codes.shape[2]
        frame = delayed_codes[..., offset : offset + 1]
        frame.masked_scatter_(frame == unknown_token, next_token)

        prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
        inference_params.seqlen_offset += prefix_length
        inference_params.lengths_per_sample[:] += prefix_length

        logit_bias = torch.zeros_like(logits)
        logit_bias[:, 1:, self.eos_token_id] = -torch.inf  # only allow codebook 0 to predict EOS

        stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_steps = delayed_codes.shape[2] - offset
        remaining_steps = torch.full((batch_size,), max_steps, device=device)
        progress = tqdm(total=max_steps, desc="Generating", disable=not progress_bar)
        cfg_scale = torch.tensor(cfg_scale)

        # H100 optimization: pre-allocate tensors
        eos_codebook_idx = torch.zeros_like(remaining_steps)

        step = 0
        while torch.max(remaining_steps) > 0:
            offset += 1
            input_ids = delayed_codes[..., offset - 1 : offset]
            logits = decode_one_token(input_ids, inference_params, cfg_scale, allow_cudagraphs=cg)
            logits += logit_bias

            next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params)
            eos_in_cb0 = next_token[:, 0] == self.eos_token_id

            remaining_steps[eos_in_cb0[:, 0]] = torch.minimum(remaining_steps[eos_in_cb0[:, 0]], torch.tensor(9))
            stopping |= eos_in_cb0[:, 0]

            eos_codebook_idx = 9 - remaining_steps
            eos_codebook_idx = torch.clamp(eos_codebook_idx, max=9 - 1)
            for i in range(next_token.shape[0]):
                if stopping[i]:
                    idx = eos_codebook_idx[i].item()
                    next_token[i, :idx] = self.masked_token_id
                    next_token[i, idx] = self.eos_token_id

            frame = delayed_codes[..., offset : offset + 1]
            frame.masked_scatter_(frame == unknown_token, next_token)
            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample[:] += 1

            remaining_steps -= 1

            progress.update()
            step += 1

            if callback is not None and not callback(frame, step, max_steps):
                break

        out_codes = revert_delay_pattern(delayed_codes)
        out_codes.masked_fill_(out_codes >= 1024, 0)
        out_codes = out_codes[..., : offset - 9]

        self._cg_graph = None  # reset cuda graph to avoid cache changes

        return out_codes

    @torch.inference_mode()
    def stream(
        self,
        prefix_conditioning: torch.Tensor,  # conditioning from text (and other modalities)
        audio_prefix_codes: torch.Tensor | None = None,  # optional audio prefix codes
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        sampling_params: dict = dict(min_p=0.1),
        disable_torch_compile: bool = False,
        chunk_schedule: list[int] = [17, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        chunk_overlap: int = 1,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Stream audio generation in chunks.
        
        Args:
            prefix_conditioning: Conditioning from text and other modalities
            audio_prefix_codes: Optional audio prefix codes
            max_new_tokens: Maximum number of new tokens to generate
            cfg_scale: Classifier-free guidance scale
            sampling_params: Parameters for sampling from logits
            disable_torch_compile: Whether to disable torch.compile
            chunk_schedule: List of chunk sizes to generate
            chunk_overlap: Number of tokens to overlap between chunks
            
        Returns:
            Generator yielding audio chunks
        """
        # H100 optimization: larger chunks for H100
        if not chunk_schedule:
            chunk_schedule = [30, *[60] * 20]  # Larger chunks for H100
            
        # H100 optimization: ensure chunk sizes are multiples of 8
        chunk_schedule = [find_multiple(cs, 8) for cs in chunk_schedule]
        
        device = self.device
        batch_size = prefix_conditioning.shape[0] // 2
        
        # Generate first chunk
        first_chunk_size = chunk_schedule[0]
        codes = self.generate(
            prefix_conditioning=prefix_conditioning,
            audio_prefix_codes=audio_prefix_codes,
            max_new_tokens=first_chunk_size,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            sampling_params=sampling_params,
            progress_bar=False,
            disable_torch_compile=disable_torch_compile,
        )
        
        # Decode and yield first chunk
        audio = self.autoencoder.decode(codes)
        yield audio
        
        # Generate remaining chunks
        total_generated = first_chunk_size
        for chunk_size in chunk_schedule[1:]:
            if total_generated >= max_new_tokens:
                break
                
            # Use the last chunk_overlap tokens as prefix for the next chunk
            prefix_len = min(chunk_overlap, codes.shape[2])
            prefix_codes = codes[..., -prefix_len:]
            
            # Generate next chunk
            next_codes = self.generate(
                prefix_conditioning=prefix_conditioning,
                audio_prefix_codes=prefix_codes,
                max_new_tokens=chunk_size,
                cfg_scale=cfg_scale,
                batch_size=batch_size,
                sampling_params=sampling_params,
                progress_bar=False,
                disable_torch_compile=disable_torch_compile,
            )
            
            # Skip the overlapping tokens to avoid duplicates
            next_codes = next_codes[..., prefix_len:]
            
            # Decode and yield next chunk
            audio = self.autoencoder.decode(next_codes)
            yield audio
            
            # Update codes and total generated
            codes = torch.cat([codes, next_codes], dim=2)
            total_generated += chunk_size - prefix_len

        self._cg_graph = None  # reset CUDA graph to avoid caching issues
