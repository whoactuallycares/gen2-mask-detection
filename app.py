from functools import partial
from pathlib import Path
from typing import List
import depthai as dai
import numpy as np


from robothub_sdk import App, IS_INTERACTIVE, CameraResolution, Config, StreamType

if IS_INTERACTIVE:
    import cv2

class Wrapper(dict):
    def __init__(self, detection):
        dict.__init__(self, detection=detection)

class MaskDetection(App):
    def on_initialize(self, unused_devices: List[dai.DeviceInfo]):
        self.msgs = {}
        self.obj_detections = []
        self.fps = 24
        self.last_state = -1
    
    def on_configuration(self, old_configuration: Config):
        print("Configuration update", self.config.values())
    
    def on_setup(self, device):

        camera = device.configure_camera(dai.CameraBoardSocket.RGB, res=CameraResolution.THE_1080_P, preview_size=(1080, 1080))
        camera.initialControl.setSceneMode(dai.CameraControl.SceneMode.FACE_PRIORITY)
        stereo = device.create_stereo_depth()  

        (face_det_nn, face_det_out, face_det_passthrough) = device.create_nn(source=device.streams.color_preview, blob_path=Path("./face-detection-retail-0004.blob"), \
        config_path=Path("./face-detection-retail-0004.json"), input_size=(300, 300), depth=stereo, nn_family="mobilenet", color_order="RGB", confidence=0.5)
        face_det_nn.setBoundingBoxScaleFactor(0.8)
        face_det_nn.setDepthLowerThreshold(100)
        face_det_nn.setDepthUpperThreshold(5000)


        (recognition_manip, recognition_manip_stream) = device.create_image_manipulator()
        recognition_manip.initialConfig.setResize(224, 224)
        recognition_manip.setWaitForConfigInput(True)


        self.script = device.create_script(script_path=Path("./script.py"),
            inputs={
                'preview': device.streams.color_preview,
                'passthrough' : face_det_passthrough,
                'face_det_in' : face_det_out
            },
            outputs={
                'manip_img': recognition_manip.inputImage,
                'manip_cfg': recognition_manip.inputConfig
            })
        
        (mask_nn, mask_nn_out, mask_nn_passthrough) = device.create_nn(source=recognition_manip_stream, blob_path=Path("./sbd_mask_classification_224x224.blob"), \
            config_path=Path("./sbd_mask_classification_224x224.json"), input_size=(224, 224))
        

        if IS_INTERACTIVE:
            device.streams.color_preview.consume(partial(self.add_msg, 'color'))
            face_det_out.consume(partial(self.add_msg, 'detection'))
            mask_nn_out.consume(partial(self.add_msg, 'recognition'))
            device.streams.synchronize((mask_nn_out, mask_nn_passthrough, device.streams.color_video), partial(self.on_detection, device.id))
        else:
            encoder = device.create_encoder(
                device.streams.color_video.output_node,
                fps=8,
                profile=dai.VideoEncoderProperties.Profile.MJPEG,
                quality=80,
            )
            encoder_stream = device.streams.create(
                encoder,
                encoder.bitstream,
                stream_type=StreamType.BINARY,
                rate=8,
            )
            device.streams.color_video.publish()
            device.streams.synchronize((mask_nn_out, mask_nn_passthrough, encoder_stream), partial(self.on_detection, device.id))

        (format_manip, format_manip_stream) = device.create_image_manipulator()
        format_manip.setMaxOutputFrameSize(1080 * 1920 * 3)
        format_manip.initialConfig.setResize([1080, 1920])
        format_manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
        format_manip.inputConfig.setWaitForMessage(True)

        self.format_script = device.create_script(script_path=Path("./format_script.py"),
            inputs={
                'frames': device.streams.color_video
            },
            outputs={
                'manip_cfg': format_manip.inputConfig,
                'manip_img': format_manip.inputImage,
            })

        
        
    def on_update(self):
        if IS_INTERACTIVE:
            for device in self.devices:
                msgs = self.get_msgs()
                if msgs is not None:
                    frame = msgs["color"].getCvFrame()
                    detections = msgs["detection"].detections
                    recognitions = msgs["recognition"]

                    for i, detection in enumerate(detections):
                        bbox = self.frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                        # Decoding of recognition results
                        rec = recognitions[i].getFirstLayerFp16()

                        index = np.argmax(self.log_softmax(rec))
                        text = "No Mask"
                        color = (0,0,255) # Red
                        if index == 1:
                            text = "Mask"
                            color = (0,255,0)

                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
                        y = (bbox[1] + bbox[3]) // 2
                        cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
                        cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)
                        coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                        cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                        cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

                    cv2.imshow("Camera", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                self.stop()

    def log_softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return np.log(e_x / e_x.sum())

    def frame_norm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def add_msg(self, name, msg : dai.NNData):
        seq = str(msg.getSequenceNum())
        if seq not in self.msgs:
            self.msgs[seq] = {}
        if 'recognition' not in self.msgs[seq]:
            self.msgs[seq]['recognition'] = []
        if name == 'recognition':
            self.msgs[seq]['recognition'].append(msg)
        elif name == 'detection':
            self.msgs[seq][name] = msg
            self.msgs[seq]["len"] = len(msg.detections)
        elif name == 'color':
            self.msgs[seq][name] = msg

    def get_msgs(self):
        seq_remove = []
        for seq, msgs in self.msgs.items():
            seq_remove.append(seq) 
            if "color" in msgs and "len" in msgs:
                if msgs["len"] == len(msgs["recognition"]):
                    for rm in seq_remove:
                        del self.msgs[rm]
                    return msgs
        return None

    def on_detection(self, device_id: str, obj_data: dai.NNData, obj_frame: dai.ImgFrame, frame: dai.ImgFrame):
        if len(obj_data.getAllLayers()) < 1:
            return

        rec = obj_data.getFirstLayerFp16()
        index = np.argmax(self.log_softmax(rec))
        if index == 1 and self.last_state != 1: # Mask
            self.send_detection(f'Still frame from device {device_id}. Subject is wearing a mask',
                tags=['detection', 'mask'], frames=[(frame, 'jpeg')])
            self.last_state = 1
        elif index == 0 and self.last_state != 0: # No mask
            self.send_detection(f'Still frame from device {device_id}. Subject is NOT wearing a mask',
                tags=['detection', 'illegal'], frames=[(frame, 'jpeg')])
            self.last_state = 0

app = MaskDetection()
app.run()
