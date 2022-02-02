import logging
from concurrent import futures
import grpc
import numpy as np
import cv2

import parsing
import vision_algorithms_pb2
import vision_algorithms_pb2_grpc


def homog_calc(request):
    """Estimates a homography that maps source_pts into dest_pts."""

    if len(request.homog_calc_args.source_pts.lines) != len(request.homog_calc_args.dest_pts.lines):
        logging.error("Source and destinations arrays have a different number of points.")

    # Build arrays from request
    src_pts = parsing.msg_to_matrix(request.homog_calc_args.source_pts)
    dst_pts = parsing.msg_to_matrix(request.homog_calc_args.dest_pts)
    ransac_thresh = request.homog_calc_args.ransac_thresh
    ransac_iters = request.homog_calc_args.max_ransac_iters

    if ransac_thresh != 0:  # Use RANSAC to estimate homography
        if ransac_thresh < 0:
            logging.error("RANSAC threshold must not be negative.")
            homog = None
        elif ransac_iters == 0:
            logging.error("If ransac_thresh is defined, then the maximum number of iterations must also be defined.")
            homog = None
        elif ransac_iters < 0:
            logging.error("RANSAC threshold must not be negative.")
            homog = None
        else:
            logging.info("Estimating homography with RANSAC.")
            homog, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thresh,
                                          maxIters=ransac_iters)
    else:  # Use least-squares method to estimate homography
        logging.info("Estimating homography without RANSAC.")
        homog, _ = cv2.findHomography(src_pts, dst_pts)

    # Return estimated homography in correct protobuf message format
    homog_msg = parsing.matrix_to_msg(homog)
    return vision_algorithms_pb2.ExecResponse(
        homog_calc_out=vision_algorithms_pb2.HomogCalcResponse(homography=homog_msg))


def homog_warp(request):
    """Applies a homography to an image and returns the resulting image."""
    # Extract arguments from request message
    width = request.homog_warp_args.out_width
    height = request.homog_warp_args.out_height
    if width is None or height is None:
        logging.error("The request must specify the output image width and height.")
    elif width <= 0 or height <= 0:
        logging.error("Output width and height must be greater than zero.")

    homography = parsing.msg_to_matrix(request.homog_warp_args.homography, n_lines=3, n_cols=3)
    image = parsing.msg_to_image(request.homog_warp_args.image)
    if homography is None or image is None:
        return vision_algorithms_pb2.ExecResponse(homog_warp_out=vision_algorithms_pb2.HomogWarpResponse())

    # Apply the homography and return transformed image
    image_warp = cv2.warpPerspective(image, homography, (width, height))
    image_msg = parsing.image_to_msg(image_warp)
    return vision_algorithms_pb2.ExecResponse(homog_warp_out=vision_algorithms_pb2.HomogWarpResponse(image=image_msg))


class VisionAlgorithmsServicer(vision_algorithms_pb2_grpc.VisionAlgorithmsServicer):
    """Provides methods that implement functionality of route guide server."""

    def Process(self, request, context):
        """Returns the output of the algorithm specified in the request."""
        if request.WhichOneof("args") is None:
            logging.error("Unknown request type.")
            return None
        elif request.WhichOneof("args") == "homog_calc_args":
            return homog_calc(request)
        elif request.WhichOneof("args") == "homog_warp_args":
            return homog_warp(request)
        # TODO other elifs


def serve():
    server = grpc.server(futures.ThreadPoolExecutor())
    vision_algorithms_pb2_grpc.add_VisionAlgorithmsServicer_to_server(
        VisionAlgorithmsServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("Successfully started and waiting for connections..")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(format='Server %(levelname)s: %(message)s', level=logging.INFO)
    serve()
