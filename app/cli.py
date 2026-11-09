import argparse, json
from .recognition import FaceService
from .processor import process_video

def main():
    parser = argparse.ArgumentParser(description="Consent-based local face matching on video")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--output", default="cli_annotated.mp4")
    args = parser.parse_args()
    service = FaceService()
    report = process_video(args.video, service, args.output)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
