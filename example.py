from generate import text_to_video

input_text = "A man climbing a cliff"
model_checkpoint = None # "combinedmodel/checkpoints/model_epoch_12.pth"
output_video_path = "output_video.mp4"

text_to_video(
    input_text=input_text,
    model_checkpoint= model_checkpoint,
    output_video_path=output_video_path,
    sequence_length=150,
    fps=15,
)
