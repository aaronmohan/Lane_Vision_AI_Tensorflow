package com.aaron.lanevisionai;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class LaneDetectorTFLite {
    private static final int GRIDING_NUM = 100;

    private Interpreter interpreter;
    public boolean ROI = false;

    public double factor = 0.0035;
    public double distToLeftLane = 0.0;
    public double distTorightLane = 0.0;
    public double betweenLanex = 0.0;

    private static final int INPUT_WIDTH = 800;
    private static final int INPUT_HEIGHT = 288;
    private static final int CHANNELS = 3;

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};
    private static final int[] TUSIMPLE_ROW_ANCHOR = {
            64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284
    };

    public LaneDetectorTFLite(AssetManager assetManager) {
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(2);
            options.setUseNNAPI(true);

            interpreter = new Interpreter(loadModelFile(assetManager), options);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private ByteBuffer loadModelFile(AssetManager assetManager) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd("lane_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private ByteBuffer preprocess(Mat input) {
        Mat rgbMat = new Mat();
        if (input.channels() == 4) {
            Imgproc.cvtColor(input, rgbMat, Imgproc.COLOR_RGBA2RGB);
        } else if (input.channels() == 1) {
            Imgproc.cvtColor(input, rgbMat, Imgproc.COLOR_GRAY2RGB);
        } else {
            input.copyTo(rgbMat);
        }

        Mat resizedMat = new Mat();
        Imgproc.resize(rgbMat, resizedMat, new Size(INPUT_WIDTH, INPUT_HEIGHT));
        rgbMat.release();

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_WIDTH * INPUT_HEIGHT * CHANNELS);
        inputBuffer.order(ByteOrder.nativeOrder());

        byte[] byteData = new byte[INPUT_WIDTH * INPUT_HEIGHT * CHANNELS];
        resizedMat.get(0, 0, byteData);
        resizedMat.release();

        for (int i = 0; i < INPUT_WIDTH * INPUT_HEIGHT; i++) {
            int rIdx = i * 3;
            int gIdx = i * 3 + 1;
            int bIdx = i * 3 + 2;

            float r = ((float) (byteData[rIdx] & 0xFF)) / 255.0f;
            float g = ((float) (byteData[gIdx] & 0xFF)) / 255.0f;
            float b = ((float) (byteData[bIdx] & 0xFF)) / 255.0f;

            r = (r - MEAN[0]) / STD[0];
            g = (g - MEAN[1]) / STD[1];
            b = (b - MEAN[2]) / STD[2];

            inputBuffer.putFloat(r);
            inputBuffer.putFloat(g);
            inputBuffer.putFloat(b);
        }

        inputBuffer.rewind();
        return inputBuffer;
    }

    private float[][][][] runInference(ByteBuffer inputBuffer) {
        float[][][][] output = new float[1][101][56][4];
        if (interpreter != null) {
            interpreter.run(inputBuffer, output);
        }
        return output;
    }

    private List<List<Point>> processOutput(float[][][][] output, int imageWidth, int imageHeight) {
        List<List<Point>> lanePoints = new ArrayList<>();
        for (int lane = 0; lane < 4; lane++) {
            lanePoints.add(new ArrayList<>());
        }

        float[][] loc = new float[56][4];

        for (int row = 0; row < 56; row++) {
            for (int lane = 0; lane < 4; lane++) {
                int maxIndex = 0;
                float maxValue = output[0][0][row][lane];

                for (int anchor = 1; anchor < 101; anchor++) {
                    float val = output[0][anchor][row][lane];
                    if (val > maxValue) {
                        maxValue = val;
                        maxIndex = anchor;
                    }
                }

                if (maxIndex == GRIDING_NUM) {
                    loc[row][lane] = -1f;
                    continue;
                }

                float threshold = 5.0f;
                if (lane == 0 || lane == 3) {
                    threshold = 2.5f;
                }

                if (maxValue < threshold) {
                    loc[row][lane] = -1f;
                    continue;
                }

                float maxLogit = Float.NEGATIVE_INFINITY;
                for (int anchor = 0; anchor < 100; anchor++) {
                    float val = output[0][anchor][row][lane];
                    if (val > maxLogit) {
                        maxLogit = val;
                    }
                }

                float sumExp = 0f;
                for (int anchor = 0; anchor < 100; anchor++) {
                    sumExp += (float) Math.exp(output[0][anchor][row][lane] - maxLogit);
                }

                if (sumExp == 0f || Float.isNaN(sumExp)) {
                    loc[row][lane] = -1f;
                    continue;
                }

                float weightedSum = 0f;
                for (int anchor = 0; anchor < 100; anchor++) {
                    float prob = (float) Math.exp(output[0][anchor][row][lane] - maxLogit) / sumExp;
                    weightedSum += prob * (anchor + 1);
                }

                loc[row][lane] = weightedSum;
            }
        }

        float colSampleW = 800.0f / GRIDING_NUM;

        for (int lane = 0; lane < 4; lane++) {
            for (int row = 0; row < 56; row++) {
                if (loc[row][lane] > 0) {
                    int x = (int) (loc[row][lane] * colSampleW * (float) imageWidth / 800.0f);
                    int y = (int) ((float) imageHeight * ((float) TUSIMPLE_ROW_ANCHOR[row] / 288.0f));

                    if (x >= 0 && x < imageWidth && y >= 0 && y < imageHeight) {
                        lanePoints.get(lane).add(new Point(x, y));
                    }
                }
            }
        }
        return lanePoints;
    }

    public void printTensorInfo() {
        if (interpreter == null) return;
        int[] inShape = interpreter.getInputTensor(0).shape();
        int[] outShape = interpreter.getOutputTensor(0).shape();

        System.out.print("Input: ");
        for (int v : inShape) {
            System.out.print(v + " ");
        }

        System.out.print("\nOutput: ");
        for (int v : outShape) {
            System.out.print(v + " ");
        }
    }

    public Mat detectLanes(Mat input, int imageHeight, int imageWidth) {
        if (input == null || input.empty()) {
            return input;
        }

        int width = (imageWidth > 0) ? imageWidth : input.cols();
        int height = (imageHeight > 0) ? imageHeight : input.rows();

        ByteBuffer inputBuffer = preprocess(input);
        float[][][][] output = runInference(inputBuffer);
        List<List<Point>> lanePoints = processOutput(output, width, height);

        Scalar[] laneColors = {
                new Scalar(255, 0, 0),   // Blue (Far left)
                new Scalar(0, 255, 0),   // Green (Ego left)
                new Scalar(0, 0, 255),   // Red (Ego right)
                new Scalar(255, 255, 0)  // Cyan (Far right)
        };

        for (int lane = 0; lane < 4; lane++) {
            List<Point> pts = lanePoints.get(lane);
            if (pts.size() > 1) {
                MatOfPoint lineMat = new MatOfPoint();
                lineMat.fromList(pts);
                List<MatOfPoint> polyList = new ArrayList<>();
                polyList.add(lineMat);
                Imgproc.polylines(input, polyList, false, laneColors[lane], 6);
            } else if (pts.size() == 1) {
                Imgproc.circle(input, pts.get(0), 5, laneColors[lane], -1);
            }
        }

        List<Point> leftLane = lanePoints.get(1);   // Green
        List<Point> rightLane = lanePoints.get(2);  // Red

        double leftBottomX = -1;
        double rightBottomX = -1;

        if (!leftLane.isEmpty()) {
            Point bottomP = leftLane.get(leftLane.size() - 1);
            leftBottomX = bottomP.x;
            distToLeftLane = Math.abs(width / 2.0 - leftBottomX) * factor;
        } else {
            distToLeftLane = 999.0;
        }

        if (!rightLane.isEmpty()) {
            Point bottomP = rightLane.get(rightLane.size() - 1);
            rightBottomX = bottomP.x;
            distTorightLane = Math.abs(rightBottomX - width / 2.0) * factor;
        } else {
            distTorightLane = 999.0;
        }

        if (leftBottomX >= 0 && rightBottomX >= 0) {
            betweenLanex = Math.abs(rightBottomX - leftBottomX) * factor;
        } else {
            betweenLanex = 0.0;
        }

        if (leftLane.size() >= 2 && rightLane.size() >= 2) {
            List<Point> polygonPoints = new ArrayList<>();
            polygonPoints.addAll(leftLane);

            for (int i = rightLane.size() - 1; i >= 0; i--) {
                polygonPoints.add(rightLane.get(i));
            }

            Mat overlay = input.clone();
            MatOfPoint poly = new MatOfPoint();
            poly.fromList(polygonPoints);

            List<MatOfPoint> polys = new ArrayList<>();
            polys.add(poly);

            Imgproc.fillPoly(overlay, polys, new Scalar(80, 180, 255));
            Core.addWeighted(overlay, 0.35, input, 0.65, 0, input);
            overlay.release();
        }

        if (ROI) {
            Point pt1 = new Point(width / 2.0, 0);
            Point pt2 = new Point(width / 2.0, height);
            Imgproc.line(input, pt1, pt2, new Scalar(0, 255, 0), 3);
        }

        return input;
    }
}
