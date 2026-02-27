from .base import track_pose

def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description='Run FoundationPose tracking with Orbbec RGB-D camera.')
    parser.add_argument('--text-prompt', type=str, default="yellow", help='Text prompt for object detection (default: yellow)')
    parser.add_argument('--mesh-file', type=str, default="tmp/scaled_mesh.obj", help='Path to the mesh file (default: tmp/scaled_mesh.obj)')
    parser.add_argument('--device-index', type=int, default=1, help='Camera device index (default: 1)')
    parser.add_argument('--server-url', type=str, default="tcp://127.0.0.1:5555", help='FoundationPose server URL (default: tcp://127.0.0.1:5555)')
    
    args = parser.parse_args()
    
    track_pose(args.text_prompt, args.mesh_file, args.device_index, args.server_url)


if __name__ == '__main__':
    run_cli()