import os
import time
import carla


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    cam_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    cam_bp.set_attribute("image_size_x", "1024")
    cam_bp.set_attribute("image_size_y", "512")
    cam_bp.set_attribute("fov", "90")
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    os.makedirs("frames", exist_ok=True)
    def callback(image):
        image.save_to_disk("frames/%06d.png" % image.frame)

    camera.listen(callback)
    vehicle.set_autopilot(True)
    time.sleep(5.0)

    camera.stop()
    camera.destroy()
    vehicle.destroy()


if __name__ == "__main__":
    main()
