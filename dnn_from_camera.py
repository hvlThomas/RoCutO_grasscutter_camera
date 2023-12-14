import pyrealsense2 as rs
import numpy as np
import cv2
import math

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Enables and sets parameters for camera stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
intrinsics = depth_profile.get_intrinsics()
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Threshold filter (x, y), filters out depth image to minimum x distance, max y distance
        threshold = rs.threshold_filter(0, 3)
        depth_threshold = threshold.process(depth_frame)
        color_threshold = threshold.process(color_frame)


        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_threshold.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.vstack((resized_color_image, depth_colormap))
        else:
            images = np.vstack((color_image, depth_colormap))

        height, width = color_image.shape[:2]
        expected = 300
        aspect = width / height
        crop_start = round(expected*(aspect-1)/2)

        net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                                       "MobileNetSSD_deploy.caffemodel")
        inScaleFactor = 0.007843
        meanVal = 127.53
        classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
                      "bottle", "bus", "car", "cat", "chair",
                      "cow", "diningtable", "dog", "horse",
                      "motorbike", "person", "pottedplant",
                      "sheep", "sofa", "train", "tvmonitor")

        blob = cv2.dnn.blobFromImage(color_image, inScaleFactor, (expected, expected), meanVal, False)
        net.setInput(blob, "data")
        detections = net.forward("detection_out")

        label = detections[0, 0, 0, 1]
        conf = detections[0, 0, 0, 2]
        xmin = detections[0, 0, 0, 3]
        ymin = detections[0, 0, 0, 4]
        xmax = detections[0, 0, 0, 5]
        ymax = detections[0, 0, 0, 6]

        className = classNames[int(label)]

        scale = height / expected
        xmin_depth = int((xmin * expected + crop_start) * scale)
        ymin_depth = int((ymin * expected) * scale)
        xmax_depth = int((xmax * expected + crop_start) * scale)
        ymax_depth = int((ymax * expected) * scale)
        xmin_depth, ymin_depth, xmax_depth, ymax_depth

        depth = np.asanyarray(depth_threshold.get_data())
        # Crop depth data:
        depth = depth[xmin_depth:xmax_depth, ymin_depth:ymax_depth].astype(float)

        # Get data scale from the device and convert to meters
        depth_scale = pipeline_profile.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale
        dist, _, _, _ = cv2.mean(depth)
        rounded_dist = round(dist, 3)

        # Calculates distance to the center of the rectangle
        x_center = (xmin + xmax) / 2 * expected
        y_center = (ymin + ymax) / 2 * expected
        z_depth = depth_frame.get_distance(int(x_center), int(y_center))
        newdist = round(z_depth, 3)

        # Calculate width of opencv rectangle
        pleft = rs.rs2_deproject_pixel_to_point(intrinsics, [xmin, y_center], newdist)
        pright = rs.rs2_deproject_pixel_to_point(intrinsics, [xmax,y_center], newdist)
        diam = round((pright[0]-pleft[0])*300,3)

        # Calculate circumference at center of cylindrical object
        rad = diam/2
        cir = round(math.pi*diam, 3)
        cdist = newdist+rad

        # Generate opencv rectangle and text to render on image
        cv2.rectangle(color_image, (xmin_depth, ymin_depth),
                      (xmax_depth, ymax_depth), (255, 255, 255), 2)
        cv2.putText(color_image, "{0} dist:{1}m, diam:{2}m circ:{3}m".format(className, newdist, diam, cir),
                    (int(xmin * expected), int(ymin * expected) - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
