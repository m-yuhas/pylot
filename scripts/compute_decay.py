from absl import app
from absl import flags
import erdust

import pylot.config
from pylot.control.messages import ControlMessage
import pylot.operator_creator
import pylot.simulation.utils

FLAGS = flags.FLAGS

flags.DEFINE_bool('compute_detection_decay', False,
                  'True to enable ground truth object detection evaluation.')
flags.DEFINE_bool('compute_segmentation_decay', False,
                  'True to enable ground truth segmentation evaluation')
flags.DEFINE_integer('decay_max_latency', 400,
                     'Max latency to evaluate in ground truth experiments')

CENTER_CAMERA_LOCATION = pylot.simulation.utils.Location(1.5, 0.0, 1.4)


class SynchronizerOperator(erdust.Operator):
    def __init__(self,
                 wait_stream,
                 control_stream,
                 flags):
        erdust.add_watermark_callback(
            [wait_stream], [control_stream], self.on_watermark)
        self._flags = flags

    @staticmethod
    def connect(wait_stream):
        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        control_stream = erdust.WriteStream()
        return [control_stream]

    def on_watermark(self, timestamp, control_stream):
        # The control message is ignored by the bridge operator because
        # data gathering is conducted using auto pilot. Send default control
        # message.
        control_msg = ControlMessage(0, 0, 0, False, False, timestamp)
        control_stream.send(control_msg)


def driver():
    transform = pylot.simulation.utils.Transform(
        CENTER_CAMERA_LOCATION, pylot.simulation.utils.Rotation(0, 0, 0))

    control_loop_stream = erdust.LoopStream()
    # Create carla operator.
    (can_bus_stream,
     ground_traffic_lights_stream,
     ground_vehicles_stream,
     ground_pedestrians_stream,
     ground_speed_limit_signs_stream,
     ground_stop_signs_stream,
     vehicle_id_stream) = pylot.operator_creator.add_carla_bridge(
         control_loop_stream)

    # Add sensors.
    (center_camera_stream,
     rgb_camera_setup) = pylot.operator_creator.add_rgb_camera(
         transform, vehicle_id_stream)
    (depth_camera_stream,
     depth_camera_setup) = pylot.operator_creator.add_depth_camera(
         transform, vehicle_id_stream)
    (segmented_stream, _) = pylot.operator_creator.add_segmented_camera(
        transform, vehicle_id_stream)

    map_stream = None
    if FLAGS.compute_detection_decay:
        obstacles_stream = pylot.operator_creator.add_perfect_detector(
            depth_camera_stream,
            center_camera_stream,
            segmented_stream,
            can_bus_stream,
            ground_pedestrians_stream,
            ground_vehicles_stream,
            ground_speed_limit_signs_stream,
            ground_stop_signs_stream,
            rgb_camera_setup)
        map_stream = pylot.operator_creator.add_detection_decay(
            obstacles_stream)

    iou_stream = None
    if FLAGS.compute_segmentation_decay:
        iou_stream = pylot.operator_creator.add_segmentation_decay(
            segmented_stream)

    # TODO: Hack! We synchronize on a single stream, based on a guesestimated
    # of which stream is slowest. Instead, We should synchronize on all output
    # streams, and we should ensure that even the operators without output
    # streams complete.
    if FLAGS.carla_auto_pilot:
        stream_to_sync_on = iou_stream
        if map_stream is not None:
            stream_to_sync_on = map_stream
        (control_stream,) = erdust.connect(
            SynchronizerOperator,
            [stream_to_sync_on],
            False,  # Does not flow watermarks.
            FLAGS)
        control_loop_stream.set(control_stream)
    else:
        raise ValueError("Must be in auto pilot mode. Pass --carla_auto_pilot")


def main(argv):
    erdust.run(driver)


if __name__ == '__main__':
    app.run(main)