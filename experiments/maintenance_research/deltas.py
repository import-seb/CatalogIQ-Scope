"""Research-only, fail-closed loading of deltas on the ORIGINAL fine-tuned base."""
from pathlib import Path
import sys
from core import FIELDS,ROOT,hash_parameters,read,sha


def validate_binding(folder,source):
    folder=Path(folder).resolve();binding=read(folder/'maintenance_manifest.json')
    if binding['base_checkpoint_sha256']!=source['files']['checkpoint.pt']:
        raise RuntimeError('Delta belongs to a different finished checkpoint')
    meta=source['metadata']
    if binding['target']!=meta['target'] or binding['label_names']!=meta['label_names'] or binding['input_fields']!=FIELDS or binding['sequence_length']!=256:
        raise RuntimeError('Delta target/vocabulary/representation differs')
    if not binding['base_is_existing_fully_fine_tuned_catalogiq_model']:
        raise RuntimeError('Replacement/pretrained-base deltas are not maintenance deltas')
    for name,expected in binding['payload_sha256'].items():
        payload=(folder/name).resolve()
        if folder not in payload.parents:raise RuntimeError('Delta payload escapes its directory')
        if sha(payload)!=expected:raise RuntimeError('Delta payload checksum mismatch')
    return binding


def load_delta(folder,source,*,device='cpu'):
    """Caller must establish the research resource limits; never infer a CUDA device."""
    folder=Path(folder);binding=validate_binding(folder,source)
    base=Path(source['directory'])
    if sha(base/'checkpoint.pt')!=binding['base_checkpoint_sha256']:raise RuntimeError('Original finished checkpoint changed')
    if sha(base/'encoder_config.json')!=source['files']['encoder_config.json']:
        raise RuntimeError('Original encoder architecture/configuration changed')
    import torch
    from transformers import AutoConfig,AutoModel
    from safetensors.torch import load_file
    sys.path.insert(0,str(ROOT/'src'))
    from catalogiq.segment_transformer import _make_classifier,_state_sha256
    config=read(base/'encoder_config.json');kind=config.pop('model_type')
    state=torch.load(base/'checkpoint.pt',map_location='cpu',weights_only=True)
    if _state_sha256(state['state_dict'])!=binding['base_final_state_sha256']:
        raise RuntimeError('Finished checkpoint tensor integrity failed')
    model=_make_classifier(torch,AutoModel.from_config(AutoConfig.for_model(kind,**config)),len(binding['label_names']),source['metadata']['config']['dropout'])
    model.load_state_dict(state['state_dict'],strict=True);del state
    if binding['method']=='lora_maintenance':
        from peft import PeftModel
        model.encoder=PeftModel.from_pretrained(model.encoder,folder/'adapter',is_trainable=False,torch_device='cpu')
        base_params=[(n.removeprefix('base_model.model.').replace('.base_layer.','.'),p) for n,p in model.encoder.named_parameters() if '.lora_A.' not in n and '.lora_B.' not in n]
    else:base_params=list(model.encoder.named_parameters())
    if hash_parameters(base_params)!=binding['fine_tuned_encoder_sha256']:
        raise RuntimeError('Delta was not attached to the original fine-tuned encoder')
    model.classifier.load_state_dict(load_file(folder/'head.safetensors',device='cpu'),strict=True)
    model.requires_grad_(False)
    return model.to(torch.device(device)).eval()
