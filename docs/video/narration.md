# NextView demo video: narration script

Spoken by a synthetic text-to-speech voice. Times are the start of each line in `nextview-demo.mp4`; the burned-in captions and `nextview-demo.srt` use the same text. Every number is read from the committed results files named in `shotlist.md`.

**0:00.5** NextView: an agent that calibrates a camera by deciding where the calibration board should go next.

**0:07.5** A solo entry by Umar Mohktar, Singapore: design, implementation and evaluation.

**0:14.2** Calibration is step zero of every metric inspection, and its usual check misleads. After a few views, the in-sample error is under a tenth of a pixel, while the true error at the image edges is many pixels.

**0:27.4** This is the live app on AWS. The camera is simulated, so the true error is known.

**0:33.2** Each step checks the frame and rejects bad ones with a measured reason, such as blur in pixels on the board's own edges, then gives a numeric instruction and a ghost outline.

**0:44.0** Candidate poses are ranked by the predicted drop in uncertainty, and each capture reports what it actually achieved.

**0:49.5** When the residuals show a radial trend, the agent asks before it changes the lens model. Deny, and the model stays pinned while the plan continues.

**1:01.8** Restart with the same seed.

**1:09.6** This time, approve, and it refits with the rational model.

**1:11.4** The race chart compares the true error with random poses on the same camera. It is one session, and the baseline keeps the k5 lens model, so part of the gap is the model switch.

**1:22.5** It stops when the calibrated chance of exceeding 0.75 pixels drops below ten percent. Export needs approval too, and returns OpenCV YAML, a report card and a JSON record.

**1:36.7** The webcam tab runs the same agent on browser frames; here, a virtual camera plays a rendered board video. Frames are never stored.

**1:50.1** It runs as one stateless Lambda container behind API Gateway, with session state in a signed token. Traces go to CloudWatch, approved exports to private S3, and a median step takes 1.4 seconds on the server.

**2:06.1** The Evidence tab serves the benchmark's figures and headline numbers, baked into the deployed image.

**2:13.0** In a pre-registered benchmark on 100 simulated test cameras, all eight primary endpoints favour the agent. After ten frames, the median true error is 1.21 pixels against 2.71 for random poses with a robot, and 1.22 against 3.61 with a human.

**2:33.0** The agent gets below one pixel in a median of 8 frames with a robot; the median random-pose run does not get there within 30.

**2:42.3** Taking parts out one at a time: the planner, outlier dropping and, on wide lenses, the model switch all help. The perception gate adds no accuracy, and costs a human 18 attempts instead of 10.

**2:55.7** The uncertainty map is optimistic, so the stop rule is calibrated on separate cameras. 190 of 300 robot runs stopped, none above 1.5 pixels, but telephoto runs never stopped.

**3:10.3** On real D455 images from the METRIC dataset, the agent is not consistently better than random frame choices: it ranks between the 2nd and 84th percentile of 100 random draws.

**3:24.2** At one metre, a 100 mm segment near the image edge is off by a median 0.047 mm after ten frames, against 0.12 to 0.17 mm for random poses. On wide lenses, though, before the lens-model switch, the agent's worst segments near the corners are off by 25.9 mm, against 5.7 mm for a dense grid.

**3:48.9** Over MCP, a local Qwen 2.5 7B model drove the same tools: 10 of 10 sessions finished, 98 percent of calls were valid, and its error matched the planner's.

**4:01.8** Failures are evidence too: across 60 fisheye, autofocus-drift and warped-board runs, the agent never issued a clean stop.

**4:10.9** NextView is decision support, not certified metrology. Primary results are simulator-graded, webcam frames are never stored, and the trace is the audit record.
