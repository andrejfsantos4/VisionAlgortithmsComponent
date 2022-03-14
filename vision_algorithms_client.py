import logging
import grpc
import numpy as np
import cv2

import vision_algorithms_pb2
import vision_algorithms_pb2_grpc
import parsing


def get_request_homog_calc(matrix1, matrix2):
    """Returns a message of type HomogCalcRequest.

    Keyword arguments:
    matrix1 -- source points to append to message
    matrix2 -- destination points to append to message
    """
    matrix1_msg = parsing.matrix_to_msg(matrix1)
    matrix2_msg = parsing.matrix_to_msg(matrix2)
    # return vision_algorithms_pb2.HomogCalcRequest(source_pts=matrix1_msg, dest_pts=matrix2_msg, ransac_thresh=5,
    #                                               max_ransac_iters=200)
    return vision_algorithms_pb2.HomogCalcRequest(source_pts=matrix1_msg, dest_pts=matrix2_msg, ransac_thresh=0,
                                                  max_ransac_iters=0)


def homog_calc_test(stub):
    # Generate ground truth homography
    angle = 0.1
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    a = 0
    b = 0
    affine = np.array([[1, a, 0],
                       [b, 1, 0],
                       [0, 0, 1]])
    c = 0
    projective = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [c, c, 1]])
    true_homog = rotation @ affine @ projective

    # Generate source and destination points, according to ground truth homography
    # src_pts = np.random.rand(2, 5) * 100
    # dst_pts = true_homog @ np.r_[src_pts, np.ones((1, src_pts.shape[1]))]
    # dst_pts[0, :] = dst_pts[0, :] / dst_pts[2, :]
    # dst_pts[1, :] = dst_pts[1, :] / dst_pts[2, :]
    # dst_pts = np.delete(dst_pts, 2, 0)

    # Hard coded points for homography
    # Camera 176 ---------------------
    # src_pts = np.array([[479, 301],
    #                     [177, 261],
    #                     [465, 139],
    #                     [227, 423]])
    # dst_pts = np.array([[240, 409],
    #                     [209, 389],
    #                     [245, 327],
    #                     [225, 428]])
    # Camera 083 ---------------------
    # src_pts = np.array([[225, 256],
    #                     [567, 270],
    #                     [634, 388],
    #                     [1175, 381]])
    # dst_pts = np.array([[317, 181],
    #                     [423, 209],
    #                     [436, 282],
    #                     [538, 279]])
    # Camera 104 ---------------------
    src_pts = np.array([[933, 598],
                        [383, 353],
                        [1075, 177],
                        [881, 131],
                        [56, 654]])

    dst_pts = np.array([[87, 469],
                        [60, 431],
                        [188, 380],
                        [149, 343],
                        [37, 466]])
    homog_request = get_request_homog_calc(src_pts, dst_pts)

    return stub.Process(vision_algorithms_pb2.ExecRequest(homog_calc_args=homog_request))


def homog_warp_test(stub, homog_msg):
    img = np.random.random_sample((300, 600, 3)) * 255
    img_msg = parsing.image_to_msg(img)
    warp_request = vision_algorithms_pb2.HomogWarpRequest(is_img=True, image=img_msg, homography=homog_msg,
                                                          out_width=10, out_height=10)
    return stub.Process(vision_algorithms_pb2.ExecRequest(homog_warp_args=warp_request))


def sift_det_test(stub):
    img = cv2.imread('car.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_msg = parsing.image_to_msg(img)

    sift_request = vision_algorithms_pb2.SiftDetRequest(image=img_msg)
    resp = stub.Process(vision_algorithms_pb2.ExecRequest(sift_det_args=sift_request))
    # keypts = parsing.msg_to_matrix(resp.sift_det_out.keypoints)
    #
    # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.drawKeypoints(gray_img, keypts, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow(img)
    # if cv2.waitKey(1000) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    return resp


def traffic_test(h, stub):
    cam = cv2.imread('delete/cam104.jpg')
    mapi = cv2.imread('delete/map_img_104.png')

    # Grid projection test
    # pts = []
    # for pt in np.ndindex(12, 7):
    #     pts.append(pt)
    # pts = np.array(pts)
    # pts[:, 0] = pts[:, 0]*100
    # pts[:, 1] = pts[:, 1]*100
    # Line projection test
    pts = np.array([[96, 391],
                    [196, 344],
                    [313, 300],
                    [371, 279],
                    [429, 257],
                    [482, 240],
                    [534, 224],
                    [580, 212],
                    [631, 195],
                    [676, 183],
                    [722, 172],
                    [764, 161],
                    [808, 149],
                    [848, 141]])

    pts_msg = parsing.matrix_to_msg(pts.T)
    homog_msg = parsing.matrix_to_msg(h)
    warp_request = vision_algorithms_pb2.HomogWarpRequest(is_img=False, points=pts_msg, homography=homog_msg)
    resp = stub.Process(vision_algorithms_pb2.ExecRequest(homog_warp_args=warp_request))
    warp_pts = parsing.msg_to_matrix(resp.homog_warp_out.points)

    counter = 0
    for warp_pt in warp_pts:
        mapi = cv2.circle(mapi, (int(warp_pt[0]), int(warp_pt[1])), radius=6, color=(counter, counter, 255), thickness=-1)
        counter += 10
    counter = 0
    for cam_pt in pts:
        cam = cv2.circle(cam, (int(cam_pt[0]), int(cam_pt[1])), radius=6, color=(counter, counter, 255), thickness=-1)
        counter += 10

    cv2.imshow('image', mapi)
    if cv2.waitKey(10000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    with grpc.insecure_channel('localhost:8061') as channel:
        estimator_stub = vision_algorithms_pb2_grpc.VisionAlgorithmsStub(channel)
        try:
            response = homog_calc_test(estimator_stub)
            homog = parsing.msg_to_matrix(response.homog_calc_out.homography)
            print("Client: Received homography ", homog)
            traffic_test(homog, estimator_stub)
            # response = homog_warp_test(estimator_stub, response.homog_calc_out.homography)
            # print("Client: Received warped image.")

            # response = sift_det_test(estimator_stub)
            # print("Client: Received feature descriptors.")
        except grpc.RpcError as rpc_error:
            print('An error has occurred:')
            print(f'  Error Code: {rpc_error.code()}')
            print(f'  Details: {rpc_error.details()}')
