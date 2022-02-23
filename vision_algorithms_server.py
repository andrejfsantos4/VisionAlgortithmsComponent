import logging
from concurrent import futures

import cv2
import grpc
import numpy as np

import parsing
import vision_algorithms_pb2
import vision_algorithms_pb2_grpc


def homog_calc(request):
    """Estimates a homography that maps source_pts into dest_pts."""

    if len(request.source_pts.lines) != len(request.dest_pts.lines):
        logging.error("Source and destinations arrays have a different number of points.")

    # Build arrays from request
    src_pts = parsing.msg_to_matrix(request.source_pts)
    dst_pts = parsing.msg_to_matrix(request.dest_pts)
    ransac_thresh = request.ransac_thresh
    ransac_iters = request.max_ransac_iters

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
    width = request.out_width
    height = request.out_height
    if width is None or height is None:
        logging.error("The request must specify the output image width and height.")
    elif width <= 0 or height <= 0:
        logging.error("Output width and height must be positive integers.")

    homography = parsing.msg_to_matrix(request.homography, n_lines=3, n_cols=3)
    image = parsing.msg_to_image(request.image)
    if homography is None or image is None:
        return vision_algorithms_pb2.ExecResponse(homog_warp_out=vision_algorithms_pb2.HomogWarpResponse())

    # Apply the homography and return transformed image
    image_warp = cv2.warpPerspective(image, homography, (width, height))
    image_msg = parsing.image_to_msg(image_warp)
    return vision_algorithms_pb2.ExecResponse(homog_warp_out=vision_algorithms_pb2.HomogWarpResponse(image=image_msg))


def sift_det(request):
    """Detects SIFT features in image and returns their positions and descriptors."""
    img = parsing.msg_to_image(request.image)
    if img is None:
        return vision_algorithms_pb2.ExecResponse(sift_det_out=vision_algorithms_pb2.SiftDetResponse())

    if request.points is not None:
        keypts = parsing.msg_to_matrix(request.points)
        if np.any((keypts < 0) | (keypts[:, 0] >= img.shape[0]) | (keypts[:, 1] >= img.shape[1])):
            logging.error("Specified keypoints have invalid coordinates.")
            return vision_algorithms_pb2.ExecResponse(sift_det_out=vision_algorithms_pb2.SiftDetResponse())
    else:
        keypts = None
    if request.n_features != 0:
        n_features = request.n_features
        if n_features < 0:
            logging.error("Specified number of features to retain must be positive integer.")
            return vision_algorithms_pb2.ExecResponse(sift_det_out=vision_algorithms_pb2.SiftDetResponse())
    else:
        n_features = 0
    if request.contrast_thresh != 0:
        contrast_thr = request.contrast_thresh
        if contrast_thr < 0:
            logging.error("Contrast threshold must be non-negative.")
            return vision_algorithms_pb2.ExecResponse(sift_det_out=vision_algorithms_pb2.SiftDetResponse())
    else:
        contrast_thr = 0.04
    if request.edge_thresh != 0:
        edge_thr = request.edge_thresh
        if edge_thr < 0:
            logging.error("Edge threshold must be non-negative.")
            return vision_algorithms_pb2.ExecResponse(sift_det_out=vision_algorithms_pb2.SiftDetResponse())
    else:
        edge_thr = 10
    if request.sigma != 0:
        sig = request.sigma
        if sig < 0:
            logging.error("Sigma must be positive.")
            return vision_algorithms_pb2.ExecResponse(sift_det_out=vision_algorithms_pb2.SiftDetResponse())
    else:
        sig = 1.6

    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_thr, edgeThreshold=edge_thr, sigma=sig)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if keypts is None:
        keypts = sift.detect(gray_img, None)
    keypts, des = sift.compute(gray_img, keypts)

    return vision_algorithms_pb2.ExecResponse(sift_det_out=vision_algorithms_pb2.SiftDetResponse(
        keypoints=parsing.array_to_msg(keypts), descriptors=parsing.array_to_msg(des)))


def sift_match(request):
    """Obtains correspondences between two sets of SIFT features descriptors."""
    des1 = parsing.msg_to_matrix(request.d1)
    des2 = parsing.msg_to_matrix(request.d2)

    # Brute Force Matcher with default params
    matcher = cv2.BFMatcher()
    aux_matches = matcher.knnMatch(des1, des2, k=2)

    # Get best matches
    matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            matches.append([m.trainIdx, m.queryIdx])
    matches_array = np.array(matches)
    #TODO finish this
    return vision_algorithms_pb2.ExecResponse(sift_match_out=vision_algorithms_pb2.SiftMatchResponse(
        matches=parsing.array_to_msg(matches_array)))


class VisionAlgorithmsServicer(vision_algorithms_pb2_grpc.VisionAlgorithmsServicer):
    """Provides methods that implement functionality of route guide server."""

    def Process(self, request, context):
        """Returns the output of the algorithm specified in the request."""
        if request.WhichOneof("args") is None:
            logging.error("Unknown request type.")
            return None
        elif request.WhichOneof("args") == "homog_calc_args":
            return homog_calc(request.homog_calc_args)
        elif request.WhichOneof("args") == "homog_warp_args":
            return homog_warp(request.homog_warp_args)
        elif request.WhichOneof("args") == "sift_det_args":
            return sift_det(request.sift_det_args)
        elif request.WhichOneof("args") == "sift_match_args":
            return sift_match(request.sift_match_args)
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
