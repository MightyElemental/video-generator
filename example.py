from generate import text_to_video

input_text = "A man climbing a cliff"
transformer_checkpoint = "text2vid/checkpoints/model_epoch_10.pth"
vae_checkpoint = "vae/checkpoints/vae_epoch_100.pth"
output_video_path = "output_video.mp4"

text_to_video(
    input_text=input_text,
    transformer_checkpoint=transformer_checkpoint,
    vae_checkpoint=vae_checkpoint,
    output_video_path=output_video_path,
    embed_dim=300,
    max_src_length=64,
    max_output_length=300,
    transformer_params={
        'embed_dim': 300,
        'vector_dim': 200,
        'nhead': 6,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'max_seq_length': 64,
        'max_output_length': 300
    },
    vae_params={
        'img_size': 256,
        'latent_dim': 200,
        'multiplier': 3
    },
    fps=30
)
