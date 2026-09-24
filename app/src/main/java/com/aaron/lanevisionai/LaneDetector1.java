package com.aaron.lanevisionai;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LaneDetector1 {
    private static final int HOUGH_THRESHOLD = 40;
    private static final int HOUGH_MIN_LINE_LENGTH = 40;
    private static final int HOUGH_MAX_LINE_GAP = 70;

    boolean ROI = false;

    double SLOPE_THRESHOLD = 0.5;
    Scalar leftColor = new Scalar(255, 0, 0);
    Scalar rightColor = new Scalar(0, 255, 0);

    // correction factor
    double factor = 0.0035;

    double distToLeftLane = 999.0;
    double distTorightLane = 999.0;
    double distToLaneCenter = 0.0;
    double betweenLanex = 0.0;
    String isOut = "";

    double minSlope = 0.5;
    double maxSlope = 3.0;

    double leftLaneX = 0.0;
    double rightLaneX = 0.0;

    public Mat detectLanes(Mat input, int imageHeight, int imageWidth) {
        if (input == null || input.empty()) {
            return input;
        }

        int height = (imageHeight > 0) ? imageHeight : input.rows();
        int width = (imageWidth > 0) ? imageWidth : input.cols();

        Mat gray = new Mat();
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGBA2GRAY);

        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);
        Imgproc.Canny(gray, gray, 150, 200);

        Mat kerneltest = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.dilate(gray, gray, kerneltest);

        Mat mask = Mat.zeros(gray.size(), gray.type());
        MatOfPoint roiPoints = new MatOfPoint(
                new Point(0, height),
                new Point(width, height),
                new Point(width * 0.6, height * 0.5),
                new Point(width * 0.4, height * 0.5)
        );

        Imgproc.fillConvexPoly(mask, roiPoints, new Scalar(255, 255, 255), Imgproc.LINE_AA);

        Mat roiEdges = new Mat();
        Core.bitwise_and(mask, gray, roiEdges);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.erode(roiEdges, roiEdges, kernel);

        Mat lines = new Mat();
        Imgproc.HoughLinesP(roiEdges, lines, 1, Math.PI / 180, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

        List<Double> rightSlope = new ArrayList<>();
        List<Double> leftSlope = new ArrayList<>();
        List<Double> rightIntercept = new ArrayList<>();
        List<Double> leftIntercept = new ArrayList<>();

        for (int i = 0; i < lines.rows(); i++) {
            double[] line = lines.get(i, 0);
            if (line == null || line.length < 4) continue;
            double x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
            if (x2 == x1) continue;

            double slope = (y2 - y1) / (x2 - x1);
            double intercept = y1 - slope * x1;

            if (slope > SLOPE_THRESHOLD) {
                rightSlope.add(slope);
                rightIntercept.add(intercept);
            } else if (slope < -SLOPE_THRESHOLD) {
                leftSlope.add(slope);
                leftIntercept.add(intercept);
            }
        }

        double leftavgSlope = leftSlope.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double leftavgIntercept = leftIntercept.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double rightavgSlope = rightSlope.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double rightavgIntercept = rightIntercept.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

        boolean leftLaneDetected = (leftavgSlope != 0 && !Double.isNaN(leftavgSlope));
        boolean rightLaneDetected = (rightavgSlope != 0 && !Double.isNaN(rightavgSlope));

        boolean validLeft = false;
        boolean validRight = false;

        Point left_p1 = null, left_p2 = null;
        Point right_p1 = null, right_p2 = null;

        if (leftLaneDetected && Math.abs(leftavgSlope) >= minSlope && Math.abs(leftavgSlope) <= maxSlope) {
            int left_x1 = (int) ((0.65 * height - leftavgIntercept) / leftavgSlope);
            int left_x2 = (int) ((height - leftavgIntercept) / leftavgSlope);
            left_p1 = new Point(left_x1, 0.65 * height);
            left_p2 = new Point(left_x2, height);

            Imgproc.line(input, left_p1, left_p2, leftColor, 10);
            leftLaneX = (left_x1 + left_x2) / 2.0;
            distToLeftLane = Math.abs(width / 2.0 - leftLaneX) * factor;
            validLeft = true;
        } else {
            distToLeftLane = 999.0;
        }

        if (rightLaneDetected && Math.abs(rightavgSlope) >= minSlope && Math.abs(rightavgSlope) <= maxSlope) {
            int right_x1 = (int) ((0.65 * height - rightavgIntercept) / rightavgSlope);
            int right_x2 = (int) ((height - rightavgIntercept) / rightavgSlope);
            right_p1 = new Point(right_x1, 0.65 * height);
            right_p2 = new Point(right_x2, height);

            Imgproc.line(input, right_p1, right_p2, rightColor, 10);
            rightLaneX = (right_x1 + right_x2) / 2.0;
            distTorightLane = Math.abs(rightLaneX - width / 2.0) * factor;
            validRight = true;
        } else {
            distTorightLane = 999.0;
        }

        if (validLeft && validRight) {
            Point[] pts = new Point[]{left_p1, left_p2, right_p2, right_p1};
            MatOfPoint points = new MatOfPoint(pts);
            Scalar color = new Scalar(0, 0, 255, 128);
            Mat overlay = new Mat(input.size(), input.type());
            input.copyTo(overlay);
            Imgproc.fillPoly(overlay, Arrays.asList(points), color);
            Core.addWeighted(overlay, 0.25, input, 0.75, 0, input);
            overlay.release();

            betweenLanex = Math.abs(rightLaneX - leftLaneX) * factor;
        } else {
            betweenLanex = 0.0;
        }

        if (ROI) {
            List<MatOfPoint> roiContours = new ArrayList<>();
            roiContours.add(roiPoints);
            Imgproc.polylines(input, roiContours, true, new Scalar(0, 255, 0), 3);

            Point pt1 = new Point(width / 2.0, 0);
            Point pt2 = new Point(width / 2.0, height);
            Imgproc.line(input, pt1, pt2, new Scalar(0, 255, 0), 3);
        }

        gray.release();
        mask.release();
        roiEdges.release();
        lines.release();

        return input;
    }
}
