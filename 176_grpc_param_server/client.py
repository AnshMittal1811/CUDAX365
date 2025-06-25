import grpc

import param_server_pb2
import param_server_pb2_grpc


def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = param_server_pb2_grpc.ParamServiceStub(channel)
    response = stub.GetParams(param_server_pb2.ParamRequest(step=1))
    print("Received params:", list(response.params))


if __name__ == "__main__":
    main()
