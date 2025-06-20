from typing import List, Generator, Optional, Tuple, Dict
from pathlib import Path
from functools import partial
import queue
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Import Hailo platform - required for operation
try:
    from hailo_platform import (HEF, VDevice,
                                FormatType, HailoSchedulingAlgorithm)
except ImportError:
    raise ImportError(
        "hailo_platform is not installed. Please install HailoRT following the setup guide:\n"
        "1. Download HailoRT from https://hailo.ai/developer-zone/software-downloads/\n"
        "2. Follow the installation guide in docs/HAILO_SETUP.md\n"
        "3. Install the Python package: pip install hailort-4.21.0-cp311-cp311-linux_aarch64.whl"
    )

IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')


class HailoAsyncInference:
    def __init__(
        self, hef_path: str, input_queue: queue.Queue,
        output_queue: queue.Queue, batch_size: int = 1,
        input_type: Optional[str] = None, output_type: Optional[Dict[str, str]] = None,
        send_original_frame: bool = False) -> None:
        """
        Initialize the HailoAsyncInference class with the provided HEF model 
        file path and input/output queues.

        Args:
            hef_path (str): Path to the HEF model file.
            input_queue (queue.Queue): Queue from which to pull input frames 
                                       for inference.
            output_queue (queue.Queue): Queue to hold the inference results.
            batch_size (int): Batch size for inference. Defaults to 1.
            input_type (Optional[str]): Format type of the input stream. 
                                        Possible values: 'UINT8', 'UINT16'.
            output_type Optional[dict[str, str]] : Format type of the output stream. 
                                         Possible values: 'UINT8', 'UINT16', 'FLOAT32'.
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.hef_path = hef_path
        self.batch_size = batch_size
        self.send_original_frame = send_original_frame
        self.output_type = output_type
        
        # Initialize Hailo device and model
        params = VDevice.create_params()
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)      
        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(output_type)

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))
    
    def _set_output_type(self, output_type_dict: Optional[Dict[str, str]] = None) -> None:
        """
        Set the output type for the HEF model. If the model has multiple outputs,
        it will set the same type for all of them.

        Args:
            output_type_dict (Optional[dict[str, str]]): Format type of the output stream.
        """
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(
                getattr(FormatType, output_type)
            )

    def callback(
        self, completion_info, bindings_list: list, input_batch: list,
    ) -> None:
        """
        Callback function for asynchronous inference.

        Args:
            completion_info: Information about the async job completion.
            bindings_list (list): List of bindings objects used in the inference.
            input_batch (list): The input batch for the inference.
        """
        # The callback function is called for each completed frame
        for i, bindings in enumerate(bindings_list):
            if completion_info.get_status() == HailoStatus.SUCCESS:
                # If the model has a single output, return the output buffer. 
                # For multi-output models like SCRFD, we also return the first output
                # as the primary detection result array for parsing.
                if len(bindings._output_names) >= 1:
                    result = bindings.output(bindings._output_names[0]).get_buffer()
                else:
                    # Fallback for unexpected cases
                    logger.warning("Model has no output names, cannot get result.")
                    result = None

                if self.send_original_frame:
                    self.output_queue.put((input_batch[i], result))
                else:
                    self.output_queue.put(result)

    def get_vstream_info(self) -> Tuple[list, list]:

        """
        Get information about input and output stream layers.

        Returns:
            Tuple[list, list]: List of input stream layer information, List of 
                               output stream layer information.
        """
        return (
            self.hef.get_input_vstream_infos(), 
            self.hef.get_output_vstream_infos()
        )

    def get_hef(self) -> HEF:
        """
        Get the object's HEF file
        
        Returns:
            HEF: A HEF (Hailo Executable File) containing the model.
        """
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the model's input layer.

        Returns:
            Tuple[int, ...]: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input

    def run(self) -> None:
        """
        Run the inference loop with real Hailo hardware.
        """
        with self.infer_model.configure() as configured_infer_model:
            job = None  # Initialize job variable
            while True:
                batch_data = self.input_queue.get()
                if batch_data is None:
                    break  # Sentinel value to stop the inference loop

                if self.send_original_frame:
                    original_batch, preprocessed_batch = batch_data
                else:
                    preprocessed_batch = batch_data

                bindings_list = []
                for frame in preprocessed_batch:
                    bindings = self._create_bindings(configured_infer_model)
                    # Ensure frame is a numpy array before setting buffer
                    if not isinstance(frame, np.ndarray):
                        frame = np.array(frame, dtype=np.uint8)
                    bindings.input().set_buffer(frame)
                    bindings_list.append(bindings)

                configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                job = configured_infer_model.run_async(
                    bindings_list, partial(
                        self.callback,
                        input_batch=original_batch if self.send_original_frame else preprocessed_batch,
                        bindings_list=bindings_list
                    )
                )
            if job is not None:
                job.wait(10000)  # Wait for the last job

    def _get_output_type_str(self, output_info) -> str:
        if self.output_type is None:
            return str(output_info.format.type).split(".")[1].lower()
        else:
            self.output_type[output_info.name].lower()

    def _create_bindings(self, configured_infer_model) -> object:
        """
        Create bindings for input and output buffers.

        Args:
            configured_infer_model: The configured inference model.

        Returns:
            object: Bindings object with input and output buffers.
        """
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=(getattr(np, self._get_output_type_str(output_info)))
                )
            for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape, 
                    dtype=(getattr(np, self.output_type[name].lower()))
                )
            for name in self.output_type
            }
        return configured_infer_model.create_bindings(
            output_buffers=output_buffers
        )


def load_images_opencv(images_path: str) -> List[np.ndarray]:
    """
    Load images from the specified path.

    Args:
        images_path (str): Path to the input image or directory of images.

    Returns:
        List[np.ndarray]: List of images as NumPy arrays.
    """
    import cv2
    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [cv2.imread(str(path))]
    elif path.is_dir():
        return [
            cv2.imread(str(img)) for img in path.glob("*")
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []

def load_input_images(images_path: str):
    """
    Load images from the specified path.

    Args:
        images_path (str): Path to the input image or directory of images.

    Returns:
        List[Image.Image]: List of PIL.Image.Image objects.
    """
    from PIL import Image
    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [Image.open(path)]
    elif path.is_dir():
        return [
            Image.open(img) for img in path.glob("*") 
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []

def validate_images(images: List[np.ndarray], batch_size: int) -> None:
    """
    Validate that images exist and are properly divisible by the batch size.

    Args:
        images (List[np.ndarray]): List of images.
        batch_size (int): Number of images per batch.

    Raises:
        ValueError: If images list is empty or not divisible by batch size.
    """
    if not images:
        raise ValueError(
            'No valid images found in the specified path.'
        )
    
    if len(images) % batch_size != 0:
        raise ValueError(
            'The number of input images should be divisible by the batch size '
            'without any remainder.'
        )


def divide_list_to_batches(
    images_list: List[np.ndarray], batch_size: int
) -> Generator[List[np.ndarray], None, None]:
    """
    Divide the list of images into batches.

    Args:
        images_list (List[np.ndarray]): List of images.
        batch_size (int): Number of images in each batch.

    Returns:
        Generator[List[np.ndarray], None, None]: Generator yielding batches 
                                                  of images.
    """
    for i in range(0, len(images_list), batch_size):
        yield images_list[i: i + batch_size]