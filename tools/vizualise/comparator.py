import os
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageComparator:
    """A class to compare images from different model outputs."""
    
    def __init__(self, directories: dict, show_image_width: int = 10):
        """
        Initialize the ImageComparator.
        
        Args:
            directories: Dictionary with model names as keys and paths as values
            show_image_width: Width for matplotlib display
        """
        self.directories = {name: Path(path) for name, path in directories.items()}
        self.show_image_width = show_image_width
        self._validate_directories()
    
    def _validate_directories(self) -> None:
        """Validate that all directories exist."""
        for name, path in self.directories.items():
            if not path.exists():
                raise FileNotFoundError(f"Directory '{name}' does not exist: {path}")
    
    def read_image(self, path: Path) -> Optional[np.ndarray]:
        """
        Read and normalize image from file.
        
        Args:
            path: Path to the image file
            
        Returns:
            Normalized image array or None if reading fails
        """
        try:
            with rasterio.open(path) as src:
                img = src.read()
                
                if img.shape[0] == 1:
                    # Grayscale image
                    img = img[0]
                elif img.shape[0] >= 3:
                    # RGB image - take first 3 bands
                    img = img[:3]
                    img = np.transpose(img, (1, 2, 0))
                else:
                    # Other cases
                    img = np.transpose(img, (1, 2, 0))
                
                # Normalize if needed
                if img.dtype == np.uint16:
                    img = (img / 65535.0 * 255).astype(np.uint8)
                elif img.dtype != np.uint8 and img.max() > 1:
                    img = (img / img.max() * 255).astype(np.uint8)
                
                return img
                
        except Exception as e:
            logger.error(f"Failed to read image {path}: {e}")
            return None
    
    def get_common_files(self) -> List[str]:
        """
        Get list of files that exist in all directories.
        
        Returns:
            List of common filenames
        """
        file_sets = []
        for name, directory in self.directories.items():
            if directory.exists():
                files = {f.name for f in directory.glob("*.TIF") if f.is_file()}
                file_sets.append(files)
                logger.info(f"Found {len(files)} .TIF files in {name} directory")
            else:
                file_sets.append(set())
        
        if not file_sets:
            return []
        
        common_files = set.intersection(*file_sets)
        logger.info(f"Found {len(common_files)} common files across all directories")
        
        return sorted(list(common_files))
    
    def calculate_figure_size(self, img_shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate optimal figure size based on image dimensions.
        
        Args:
            img_shape: (height, width) of the image
            
        Returns:
            (width, height) for matplotlib figure
        """
        height, width = img_shape
        aspect_ratio = width / height
        fig_height = len(self.directories) * self.show_image_width / aspect_ratio
        return self.show_image_width, fig_height
    
    def display_comparison(self, filename: str, save_path: Optional[Path] = None) -> bool:
        """
        Display comparison of a single image across all models.
        
        Args:
            filename: Name of the file to compare
            save_path: Optional path to save the comparison plot
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\nDisplaying: {filename}")
        
        # Load images
        images = {}
        for name, directory in self.directories.items():
            img_path = directory / filename
            img = self.read_image(img_path)
            if img is None:
                logger.warning(f"Could not load image from {name}: {img_path}")
                return False
            images[name] = img
        
        if not images:
            logger.error(f"No images loaded for {filename}")
            return False
        
        # Get image dimensions (assume all images have same dimensions)
        first_img = next(iter(images.values()))
        fig_width, fig_height = self.calculate_figure_size(first_img.shape[:2])
        
        # Create subplot
        n_models = len(images)
        fig, axes = plt.subplots(n_models, 1, figsize=(fig_width, fig_height))
        
        # Handle single subplot case
        if n_models == 1:
            axes = [axes]
        
        # Display images
        for ax, (model_name, img) in zip(axes, images.items()):
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                # Ensure RGB values are in correct range
                if img.max() <= 1.0:
                    ax.imshow(img)
                else:
                    ax.imshow(img.astype(np.uint8))
            
            ax.set_title(f"{model_name}: {filename}", fontsize=12, pad=10)
            ax.axis("off")
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            save_file = save_path / f"{filename}_comparison.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison to {save_file}")
        
        # Comment out or remove image display
        # plt.show()
        return True
    
    def run_comparison(self, limit: Optional[int] = None, 
                      save_comparisons: bool = False, 
                      output_dir: Optional[Path] = None) -> None:
        """
        Run comparison for all common files with optional limit.
        
        Args:
            limit: Maximum number of images to process (None for all)
            save_comparisons: Whether to save comparison plots
            output_dir: Directory to save plots (required if save_comparisons=True)
        """
        if save_comparisons and output_dir is None:
            raise ValueError("output_dir must be specified if save_comparisons=True")
        
        if save_comparisons:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        common_files = self.get_common_files()
        
        if not common_files:
            logger.error("No common files found across all directories")
            return
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            common_files = common_files[:limit]
            print(f"Processing first {len(common_files)} files (limit applied)")
        else:
            print(f"Processing all {len(common_files)} files")
        
        for i, filename in enumerate(common_files, 1):
            print(f"\n[{i}/{len(common_files)}] Processing: {filename}")
            
            success = self.display_comparison(
                filename, 
                save_path=output_dir if save_comparisons else None
            )
            
            if not success:
                print(f"Skipped {filename} due to errors")
        
        print("Comparison complete!")

def run_comparison_from_config(directories, output_dir, show_image_width=10, limit=None, save_comparisons=True):
    comparator = ImageComparator(directories, show_image_width)
    comparator.run_comparison(
        limit=limit,
        save_comparisons=save_comparisons,
        output_dir=output_dir if save_comparisons else None
    )

def main():
    """Main function to run the image comparison."""
    
    # Configuration
    directories = {
        "Pre-trained": r"D:\Sagi\GCP\GCP\work_dirs\gcp_r50_kazgisa-kostanai\gcp-r50-pretrained-test-v1\results",
        "Fine-tuned": r"D:\Sagi\GCP\GCP\work_dirs\gcp_r50_kazgisa-kostanai\mask2former-finetune-7e-test-v1\results",
    }
    
    show_image_width = 10
    save_comparisons = True  # ✅ Enabled saving
    output_dir = Path("comparison_outputs")  # Directory to save plots
    limit = 30  # Set limit for number of images to process (None for all)
    
    try:
        # Suppress rasterio warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")
        
        # Create comparator and run
        comparator = ImageComparator(directories, show_image_width)
        comparator.run_comparison(
            limit=limit,
            save_comparisons=save_comparisons,
            output_dir=output_dir if save_comparisons else None
        )
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
