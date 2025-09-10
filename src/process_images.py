#!/usr/bin/env python3
"""
CLI tool for processing pavement cell images using ONNX Runtime.

This script processes images from a specified folder and generates segmentation masks
using a pre-trained U-Net model converted to ONNX format.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import logging

import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TILE_SIZE = 256  # Fixed tile size for processing

class PavementCellProcessor:
    """Process pavement cell images using ONNX Runtime."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """Initialize the processor with ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.session = self._load_model()
        
    def _load_model(self) -> ort.InferenceSession:
        """Load ONNX model and create inference session."""
        try:
            # Set up providers based on device
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                logger.info("Using CUDA for inference")
            else:
                logger.info("Using CPU for inference")
            
            session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info(f"Successfully loaded ONNX model from {self.model_path}")
            return session
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            sys.exit(1)
    
    def _pad_image(self, image: Image.Image, overlap: int) -> Tuple[Image.Image, int, int]:
        """Pad image to make it divisible into tiles.
        
        Args:
            image: Input PIL Image
            overlap: Overlap between tiles
            
        Returns:
            Tuple of (padded_image, pad_width, pad_height)
        """
        width, height = image.size
        pad_height = (TILE_SIZE - (height % (TILE_SIZE - overlap))) % (TILE_SIZE - overlap)
        pad_width = (TILE_SIZE - (width % (TILE_SIZE - overlap))) % (TILE_SIZE - overlap)
        
        logger.debug(f"Original image size: {width}x{height}, Padding: {pad_width}x{pad_height}")
        
        # Pad image using reflect mode
        padded_image = Image.new(image.mode, (width + pad_width, height + pad_height))
        padded_image.paste(image, (0, 0))
        
        # Fill padded areas by reflecting edges
        if pad_width > 0:
            # Right edge reflection
            for x in range(width, width + pad_width):
                for y in range(height):
                    padded_image.putpixel((x, y), image.getpixel((width - 1 - (x - width), y)))
        
        if pad_height > 0:
            # Bottom edge reflection
            for y in range(height, height + pad_height):
                for x in range(width + pad_width):
                    ref_y = height - 1 - (y - height)
                    padded_image.putpixel((x, y), padded_image.getpixel((x, ref_y)))
        
        return padded_image, pad_width, pad_height
    
    def _preprocess_tile(self, tile: Image.Image) -> np.ndarray:
        """Preprocess a tile for model inference.
        
        Args:
            tile: PIL Image tile
            
        Returns:
            Preprocessed numpy array ready for inference
        """
        # Ensure tile is the correct size
        if tile.size != (TILE_SIZE, TILE_SIZE):
            # Pad tile to TILE_SIZE x TILE_SIZE
            padded_tile = Image.new('RGB', (TILE_SIZE, TILE_SIZE), (0, 0, 0))
            padded_tile.paste(tile, (0, 0))
            tile = padded_tile
        
        # Convert to numpy array and normalize
        tile_array = np.array(tile).astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format (channels first)
        tile_array = np.transpose(tile_array, (2, 0, 1))
        
        # Add batch dimension
        tile_array = np.expand_dims(tile_array, axis=0)
        
        return tile_array
    
    def predict_mask_full_image(self, image: Image.Image, overlap: int = 64, 
                              threshold: float = 0.9, save_debug: bool = False, 
                              output_dir: Optional[str] = None, debug_subdir: Optional[str] = None,
                              use_intersection_blend: bool = True) -> np.ndarray:
        """Predict segmentation mask for full image using tiled inference.
        
        Args:
            image: Input PIL Image
            overlap: Overlap between adjacent tiles
            threshold: Threshold for binary segmentation
            save_debug: Whether to save debug visualizations
            output_dir: Directory to save debug images
            debug_subdir: Subdirectory path for debug files (preserves input structure)
            use_intersection_blend: Whether to use intersection blending for overlaps.
                                  If True, only pixels where ALL overlapping tiles agree are kept.
                                  If False, uses averaging blending (legacy behavior).
            
        Returns:
            Segmentation mask as numpy array
        """
        # Pad image
        padded_image, pad_width, pad_height = self._pad_image(image, overlap)
        original_width, original_height = image.size
        
        logger.debug(f"Padded image size: {padded_image.width}x{padded_image.height}")
        
        # Initialize output arrays
        full_mask = np.zeros((padded_image.height, padded_image.width))
        count_map = np.zeros((padded_image.height, padded_image.width))
        
        tile_count = 0
        
        # Process tiles
        for y in range(0, padded_image.height - overlap, TILE_SIZE - overlap):
            for x in range(0, padded_image.width - overlap, TILE_SIZE - overlap):
                tile_count += 1
                
                # Extract tile
                tile = padded_image.crop((
                    x, y,
                    min(x + TILE_SIZE, padded_image.width),
                    min(y + TILE_SIZE, padded_image.height)
                ))
                
                # Preprocess tile
                tile_array = self._preprocess_tile(tile)
                
                # Run inference
                input_name = self.session.get_inputs()[0].name
                output_name = self.session.get_outputs()[0].name
                
                outputs = self.session.run([output_name], {input_name: tile_array})
                tile_mask_logits = outputs[0].squeeze()
                
                # Apply sigmoid and threshold
                tile_mask = 1 / (1 + np.exp(-tile_mask_logits))  # Sigmoid
                tile_mask = (tile_mask > threshold).astype(np.float32)
                tile_mask = 1 - tile_mask  # Invert mask
                
                # Resize mask back to original tile size if padding was added
                tile_mask_resized = tile_mask[:tile.height, :tile.width]
                
                # Update full mask and count map
                full_mask[y:y + tile.height, x:x + tile.width] += tile_mask_resized
                count_map[y:y + tile.height, x:x + tile.width] += 1
        
        # Blend overlapping areas based on the selected method
        if use_intersection_blend:
            # Intersection-based blending for overlaps
            # For non-overlapping areas (count_map == 1): keep original value
            # For overlapping areas (count_map > 1): only keep pixels where ALL tiles agree (intersection)
            intersection_mask = np.zeros_like(full_mask)
            
            # Non-overlapping areas: use original mask value
            non_overlap_mask = (count_map == 1)
            intersection_mask[non_overlap_mask] = full_mask[non_overlap_mask]
            
            # Overlapping areas: intersection logic
            overlap_mask = (count_map > 1)
            # For intersection: all tiles must have mask value 1, so sum equals count
            intersection_condition = (full_mask == count_map) & overlap_mask
            intersection_mask[intersection_condition] = 1.0
            
            full_mask = intersection_mask
        else:
            # Legacy averaging blend method
            full_mask = np.divide(full_mask, count_map, out=np.zeros_like(full_mask), where=count_map!=0)
        
        # Crop out padding to get final mask
        final_mask = full_mask[:original_height, :original_width]
        
        logger.info(f"Processed {tile_count} tiles")
        
        # Save debug visualization if requested
        if save_debug and output_dir:
            if debug_subdir:
                debug_path = Path(output_dir) / debug_subdir / "debug"
            else:
                debug_path = Path(output_dir) / "debug"
            debug_path.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(final_mask, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(image)
            plt.imshow(final_mask, alpha=0.5, cmap='Reds')
            plt.title("Overlay")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(debug_path / f"debug_{Path(image.filename).stem}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        return final_mask
    
    def process_folder(self, input_folder: str, output_folder: str, 
                      overlap: int = 64, threshold: float = 0.9, 
                      save_debug: bool = False, use_intersection_blend: bool = True) -> None:
        """Process all images in a folder.
        
        Args:
            input_folder: Path to folder containing input images
            output_folder: Path to folder for saving output masks
            overlap: Overlap between adjacent tiles
            threshold: Threshold for binary segmentation
            save_debug: Whether to save debug visualizations
            use_intersection_blend: Whether to use intersection blending for overlaps
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files (including in subdirectories)
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = [
            f for f in input_path.rglob('*') 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.warning(f"No image files found in {input_folder}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                # Load image
                image = Image.open(image_file).convert('RGB')
                logger.debug(f"Processing {image_file.name} ({image.size})")
                
                # Store filename for debug visualization
                image.filename = str(image_file)
                
                # Calculate relative path for output structure
                relative_path = image_file.relative_to(input_path)
                debug_subdir = str(relative_path.parent) if relative_path.parent != Path('.') else None
                
                # Predict mask
                mask = self.predict_mask_full_image(
                    image, overlap, threshold, save_debug, str(output_path), debug_subdir, use_intersection_blend
                )
                
                # Create relative path structure in output
                output_subdir = output_path / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                # Save mask
                mask_filename = output_subdir / f"{image_file.stem}_mask.png"
                mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                mask_image.save(mask_filename)
                
                logger.info(f"Saved mask: {mask_filename}")
                
            except Exception as e:
                logger.error(f"Failed to process {image_file.name}: {e}")
                continue


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Process pavement cell images using ONNX Runtime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_folder",
        help="Path to folder containing input images"
    )
    
    parser.add_argument(
        "output_folder", 
        help="Path to folder for saving output masks"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="models/unet_model.onnx",
        help="Path to ONNX model file"
    )
    
    parser.add_argument(
        "--device", "-d",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for inference"
    )
    
    
    parser.add_argument(
        "--overlap", "-o",
        type=int,
        default=64,
        help="Overlap between tiles"
    )
    
    parser.add_argument(
        "--threshold", "-th",
        type=float,
        default=0.9,
        help="Threshold for binary segmentation"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug visualizations"
    )
    
    parser.add_argument(
        "--intersection-blend",
        action="store_true",
        default=True,
        help="Use intersection blending for overlaps (default: True). Only pixels where ALL overlapping tiles agree are kept."
    )
    
    parser.add_argument(
        "--average-blend",
        action="store_true",
        help="Use legacy averaging blending for overlaps instead of intersection blending"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine blend method
    use_intersection_blend = not args.average_blend
    
    # Validate inputs
    if not Path(args.input_folder).exists():
        logger.error(f"Input folder does not exist: {args.input_folder}")
        sys.exit(1)
    
    if not Path(args.model).exists():
        logger.error(f"Model file does not exist: {args.model}")
        sys.exit(1)
    
    # Initialize processor
    processor = PavementCellProcessor(args.model, args.device)
    
    # Process images
    processor.process_folder(
        args.input_folder,
        args.output_folder,
        args.overlap,
        args.threshold,
        args.debug,
        use_intersection_blend
    )
    
    logger.info("Processing completed!")


if __name__ == "__main__":
    main()
