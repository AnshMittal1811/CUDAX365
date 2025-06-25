import time
import concurrent.futures

import grpc

import param_server_pb2
import param_server_pb2_grpc


class ParamService(param_server_pb2_grpc.ParamServiceServicer):
    def GetParams(self, request, context):
        params = [0.1 * (request.step + i) for i in range(8)]
        return param_server_pb2.ParamResponse(params=params)


def serve():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=2))
    param_server_pb2_grpc.add_ParamServiceServicer_to_server(ParamService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
