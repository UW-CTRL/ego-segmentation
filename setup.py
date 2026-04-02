from setuptools import setup, find_packages

setup(
    name="egocentric_segmentation",
    version="0.1.0",
    description="Egocentric Video Segmentation using SAM2 and YOLOv5",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "Pillow",
        "matplotlib",
        "opencv-python"
    ],
    python_requires=">=3.8",
)