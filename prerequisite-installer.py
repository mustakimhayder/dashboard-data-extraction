import sys
import subprocess
import platform
import os
from typing import List, Tuple
import time

class PrerequisiteInstaller:
    def __init__(self):
        self.bootstrap_packages = ['setuptools', 'wheel']
        # Define installation order and package groups
        self.installation_order = [
            {
                'name': 'Initial EasyOCR Installation',
                'packages': ['easyocr'],
                'post_install': self.handle_opencv_conflict
            },
            {
                'name': 'Core Dependencies',
                'packages': [
                    'numpy',
                    'pillow',
                ],
                'post_install': None
            },
            {
                'name': 'GUI Dependencies',
                'packages': ['tk'],  # Usually included with Python, but listed for verification
                'post_install': None
            }
        ]

    def bootstrap(self):
        """Install basic required packages for the installer to run"""
        print("Installing bootstrap dependencies...")
        try:
            for package in self.bootstrap_packages:
                subprocess.check_call([
                    sys.executable,
                    '-m',
                    'pip',
                    'install',
                    '--upgrade',
                    package
                ])
            # Now we can import pkg_resources
            global pkg_resources
            import pkg_resources
            return True
        except Exception as e:
            print(f"Error during bootstrap: {e}")
            return False

    def check_python_version(self) -> bool:
        """Check if Python version meets minimum requirements"""
        min_version = (3, 7)
        current_version = sys.version_info[:2]
        return current_version >= min_version

    def handle_opencv_conflict(self):
        """Handle the OpenCV conflict after EasyOCR installation"""
        print("\nHandling OpenCV dependencies...")
        try:
            # Uninstall any existing OpenCV versions
            opencv_packages = [
                'opencv-python-headless',
                'opencv-python',
                'opencv-contrib-python-headless',
                'opencv-contrib-python'
            ]
            
            for package in opencv_packages:
                try:
                    print(f"Removing {package} if installed...")
                    subprocess.check_call([
                        sys.executable,
                        '-m',
                        'pip',
                        'uninstall',
                        '-y',
                        package
                    ])
                except subprocess.CalledProcessError:
                    pass  # Package might not be installed
            
            # Install the specific OpenCV version needed
            print("Installing opencv-python for GUI support...")
            subprocess.check_call([
                sys.executable,
                '-m',
                'pip',
                'install',
                '--no-cache-dir',  # Avoid using cached wheels
                'opencv-python'
            ])
            
            # Small delay to ensure installation is complete
            time.sleep(2)
            
        except subprocess.CalledProcessError as e:
            print(f"Error handling OpenCV dependencies: {e}")
            return False
        return True

    def install_system_dependencies(self):
        """Install system-level dependencies based on OS"""
        system = platform.system().lower()
        
        if system == 'linux':
            dependencies = [
                'python3-tk',
                'python3-pil',
                'python3-pil.imagetk',
                'libgl1-mesa-glx'  # Required for OpenCV GUI
            ]
            
            try:
                print("Installing system dependencies...")
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                subprocess.run(['sudo', 'apt-get', 'install', '-y'] + dependencies, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error installing system dependencies: {e}")
                print("Please install the following packages manually:")
                print("\n".join(dependencies))

    def get_installed_packages(self):
        """Get list of installed packages"""
        try:
            import pkg_resources
            return {pkg.key for pkg in pkg_resources.working_set}
        except ImportError:
            return set()

    def install_package_group(self, group):
        """Install a group of packages and run post-install handler if specified"""
        print(f"\nInstalling {group['name']}...")
        
        for package in group['packages']:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable,
                    '-m',
                    'pip',
                    'install',
                    '--upgrade',
                    package
                ])
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                return False

        if group['post_install']:
            return group['post_install']()
        return True

    def verify_installations(self) -> Tuple[bool, List[str]]:
        """Verify all required packages are installed correctly"""
        required_imports = {
            'easyocr': 'easyocr',
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'PIL': 'pillow',
            'tkinter': 'tk'
        }
        
        failed_imports = []
        for import_name, package_name in required_imports.items():
            try:
                __import__(import_name)
                if import_name == 'cv2':
                    # Verify OpenCV GUI support
                    import cv2
                    if not hasattr(cv2, 'imshow'):
                        failed_imports.append(f"{package_name} (GUI support missing)")
            except ImportError:
                failed_imports.append(package_name)
        
        return len(failed_imports) == 0, failed_imports

    def run(self):
        """Run the complete installation process"""
        print("Starting prerequisite installation...")
        print("=" * 50)
        
        # Bootstrap essential packages
        if not self.bootstrap():
            print("Failed to install essential dependencies. Please install setuptools manually:")
            print("pip install setuptools wheel")
            return False

        # Check Python version
        if not self.check_python_version():
            print("Error: Python 3.7 or higher is required")
            return False

        # Install system dependencies if needed
        if platform.system().lower() == 'linux':
            self.install_system_dependencies()

        # Install package groups in order
        for group in self.installation_order:
            if not self.install_package_group(group):
                print(f"\nError during {group['name']} installation")
                return False

        # Verify installations
        print("\nVerifying installations...")
        success, failed = self.verify_installations()
        if success:
            print("\nAll prerequisites installed successfully!")
            print("\nImportant: If you experience any GUI-related issues:")
            print("1. Try restarting your Python environment")
            print("2. Verify that opencv-python (not headless) is installed:")
            print("   pip show opencv-python")
            return True
        else:
            print("\nThe following packages failed to install correctly:")
            print("\n".join(failed))
            print("\nPlease install them manually or check system requirements.")
            return False

def main():
    print("Prerequisite Installer for Image Processing Tools")
    print("=" * 50)
    
    installer = PrerequisiteInstaller()
    if installer.run():
        print("\nSetup complete! You can now run the image processing scripts.")
    else:
        print("\nSetup incomplete. Please resolve the above issues before running the scripts.")

if __name__ == "__main__":
    main()
