# KTT tutorial demonstrating how the tuner is initialized.
# Serves more as a sanity check that KTT is working correctly.
# Users are recommended to start with KTT Introductory guide
# at https://github.com/HiPerCoRe/KTT/blob/master/OnboardingGuide.md
# before reading the tutorial's code.

import sys
import pyktt as ktt

def main():
    platformIndex = 0
    deviceIndex = 0

    argc = len(sys.argv)

    if argc >= 2:
        platformIndex = int(sys.argv[1])

        if argc >= 3:
            deviceIndex = int(sys.argv[2])

    # Create new tuner which uses CUDA as compute API.
    tuner = ktt.Tuner(platformIndex, deviceIndex, ktt.ComputeApi.CUDA)
    deviceInfo = tuner.GetCurrentDeviceInfo()
    print("Current device info: " + deviceInfo.GetString())

    print("Tuner successfully initialized.")

if __name__ == "__main__":
    main()
