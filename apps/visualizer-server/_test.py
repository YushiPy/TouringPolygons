
import os

from PIL import Image

def depixelize(image_path: str, scale: int, output_path: str) -> None:
	
	img = Image.open(image_path).convert("RGBA")
	w, h = img.size

	target_width  = int(w / scale)
	target_height = int(h / scale)

	result = Image.new("RGBA", (target_width, target_height))

	for row in range(target_height):
		for col in range(target_width):
			sample_x = int((col + 0.5) * scale)
			sample_y = int((row + 0.5) * scale)

			sample_x = min(sample_x, w - 1)
			sample_y = min(sample_y, h - 1)

			pixel: tuple[int, int, int, int] = img.getpixel((sample_x, sample_y)) # type: ignore

			# If alpha  is almost 0, make it fully transparent
			if pixel[3] < 10:
				result.putpixel((col, row), (0, 0, 0, 0))
			else:
				result.putpixel((col, row), pixel)

	result.save(output_path)
	print(f"Saved {target_width}×{target_height} image to '{output_path}'.")

def depixelize_dir(
	input_dir: str,
	scale: int,
	output_dir: str,
):
	import os

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for filename in os.listdir(input_dir):
		if filename.lower().endswith((".png", ".jpg", ".jpeg")):
			input_path = os.path.join(input_dir, filename)
			output_path = os.path.join(output_dir, filename)
			depixelize(input_path, scale, output_path)

if __name__ == "__main__":

	RATIO = 21

	input_dir = "/Users/gabrielushijima/Downloads/patterns"
	output_dir = "output_images"

	depixelize_dir(input_dir, RATIO, output_dir)
