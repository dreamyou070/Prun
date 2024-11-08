import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def main() :

    print(f' step 1. get original model')
    device = "cuda"
    dtype = torch.float16
    step = 4
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    base = "emilianJR/epiCRealism"  # Choose to your favorite base model.
    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

    print(f' step 2. check blocks')
    unet = pipe.unet
    for net_name, net in unet.named_children():
        print(f' net_name: {net_name}, net: {net}')
    """
    def register_motion_editor(unet, editor: AttentionBase):
        def register_editor(net, count, place_in_unet, net_name):

            for name, subnet in net.named_children():
                final_name = f"{net_name}_{name}"

                # train only this module
                
                if subnet.__class__.__name__ == 'TransformerTemporalModel' or subnet.__class__.__name__ == 'AnimateDiffTransformer3D':
                    if final_name in editor.skip_layers:

                        if not editor.is_teacher :
                            basic_dim = subnet.proj_in.in_features
                            simple_block = SimpleAttention(basic_dim, layer_name=final_name)
                            setattr(net, name, simple_block)
                            subnet = simple_block

                        else:
                            subnet.forward = motion_forward_basic(subnet, final_name)


                    # caching the output (only the
                # if subnet.__class__.__name__ == 'BasicTransformerBlock' and 'motion' in final_name.lower():
                #    subnet.forward = motion_forward_basictransformerblock(subnet, final_name)

                # if subnet.__class__.__name__ == 'FeedForward' and 'motion' in final_name.lower():
                #    subnet.forward = motion_feedforward(subnet, final_name)

                if subnet.__class__.__name__ == 'Attention' and 'motion' not in final_name.lower():
                    subnet.forward = attention_forward(subnet, final_name)

                # if subnet.__class__.__name__ == 'Attention' and 'motion' in final_name.lower():
                #     subnet.forward = motion_forward(subnet, final_name)  # attention again

                if hasattr(net, 'children'):
                    count = register_editor(subnet, count, place_in_unet, final_name)

            return count

        cross_att_count = 0
        for net_name, net in unet.named_children():
            if "down" in net_name:
                cross_att_count += register_editor(net, 0, "down", net_name)
            elif "mid" in net_name:
                cross_att_count += register_editor(net, 0, "mid", net_name)
            elif "up" in net_name:
                cross_att_count += register_editor(net, 0, "up", net_name)
        editor.num_att_layers = cross_att_count
    """

if __name__ == '__main__':
    main()