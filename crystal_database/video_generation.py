import cv2
import os
import numpy as np

def create_video_from_images(image_folder, output_path, output_file, fps=10):  # Frame rate: 10
    # Get all image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort(reverse=True)  # Sort images in reverse order by filename

    if not images:
        print("No image files found!")
        return

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

    # Get the size of the first image
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        print(f"Failed to read image: {first_image_path}")
        return

    height, width = frame.shape[:2]
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    full_output_path = os.path.join(output_path, output_file)
    video = cv2.VideoWriter(full_output_path, fourcc, fps, (width, height))

    # Iterate through all images and write them to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if frame is None:
            print(f"Failed to read image: {image_path}")
            continue

        # Check if the image has an alpha channel
        if frame.shape[2] == 4:
            # Create a black background
            black_background = np.zeros((height, width, 3), dtype=np.uint8)
            # Extract the alpha channel and other color channels
            alpha_channel = frame[:, :, 3] / 255.0
            color_channels = frame[:, :, :3]

            # Combine the image with the black background
            for c in range(3):
                black_background[:, :, c] = alpha_channel * color_channels[:, :, c] + (
                            1 - alpha_channel) * black_background[:, :, c]
        else:
            # If no alpha channel, use the image directly
            black_background = frame

        video.write(black_background)

    # Release the video object
    video.release()
    print(f"Video has been saved to {full_output_path}")

# Example usage
image_folder = r'image_temp'  # Replace with the path to your image folder
output_path = r'video_temp'  # Replace with the path to the folder where the video will be saved
output_file = 'cubic_432_2_-2.mp4'  # Output video file name
create_video_from_images(image_folder, output_path, output_file)
