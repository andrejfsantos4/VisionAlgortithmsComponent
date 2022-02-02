# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import vision_algorithms_pb2 as vision__algorithms__pb2


class VisionAlgorithmsStub(object):
    """Service that receives a request, executes requested algorithm on that image
    and returns the corresponding output
    :param ExecRequest: The request specifying the algorithm and its inputs
    :returns: The corresponding output from the algorithm
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Process = channel.unary_unary(
                '/VisionAlgorithms/Process',
                request_serializer=vision__algorithms__pb2.ExecRequest.SerializeToString,
                response_deserializer=vision__algorithms__pb2.ExecResponse.FromString,
                )


class VisionAlgorithmsServicer(object):
    """Service that receives a request, executes requested algorithm on that image
    and returns the corresponding output
    :param ExecRequest: The request specifying the algorithm and its inputs
    :returns: The corresponding output from the algorithm
    """

    def Process(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_VisionAlgorithmsServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Process': grpc.unary_unary_rpc_method_handler(
                    servicer.Process,
                    request_deserializer=vision__algorithms__pb2.ExecRequest.FromString,
                    response_serializer=vision__algorithms__pb2.ExecResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'VisionAlgorithms', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class VisionAlgorithms(object):
    """Service that receives a request, executes requested algorithm on that image
    and returns the corresponding output
    :param ExecRequest: The request specifying the algorithm and its inputs
    :returns: The corresponding output from the algorithm
    """

    @staticmethod
    def Process(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/VisionAlgorithms/Process',
            vision__algorithms__pb2.ExecRequest.SerializeToString,
            vision__algorithms__pb2.ExecResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)