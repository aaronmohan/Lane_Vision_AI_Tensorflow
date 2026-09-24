package com.aaron.lanevisionai;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.provider.MediaStore;
import android.text.SpannableString;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.WindowManager;
import android.view.animation.AlphaAnimation;
import android.view.animation.Animation;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

public class videoClass extends CameraActivity {

    Button detect, detectfile;
    ImageButton soundon, soundoff;

    CheckBox checkBox, checkBoxLane;
    ImageView imageViewAlertR, imageViewAlertL;
    ImageView videoView;

    CameraBridgeViewBase cameraBridgeViewBase;

    LaneDetectorTFLite laneDetector;
    private int frameCounter = 0;
    private Mat lastResult = null;
    EditText factorDisplay;
    TextView textViewTest;

    String factorDisplayString;
    Double factorDouble;

    boolean isAssist = false;
    private Context context;
    private boolean isSoundOn = true;
    boolean isRoutineRunning = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.roadvideo_land);

        context = this;

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (OpenCVLoader.initDebug()) {
            Log.d("LOADED", "Success");
        } else {
            Log.d("LOADED", "error");
        }

        getPermission();

        soundon = findViewById(R.id.soundon);
        soundoff = findViewById(R.id.soundoff);

        imageViewAlertR = findViewById(R.id.imageViewAlertR);
        imageViewAlertL = findViewById(R.id.imageViewAlertL);

        imageViewAlertR.setVisibility(View.GONE);
        imageViewAlertL.setVisibility(View.GONE);

        textViewTest = findViewById(R.id.textViewTest);
        textViewTest.setText("");

        detect = findViewById(R.id.detect);
        detectfile = findViewById(R.id.detectfile);
        checkBox = findViewById(R.id.checkBox);
        checkBoxLane = findViewById(R.id.checkBoxLane);

        cameraBridgeViewBase = findViewById(R.id.cameraView);
        cameraBridgeViewBase.setVisibility(View.GONE);

        videoView = findViewById(R.id.videoView);
        videoView.setVisibility(View.GONE);

        factorDisplay = findViewById(R.id.factor);
        factorDisplay.setText("0.0035");

        laneDetector = new LaneDetectorTFLite(getAssets());
        laneDetector.printTensorInfo();

        factorDisplay.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i == EditorInfo.IME_ACTION_DONE) {
                    factorDisplayString = factorDisplay.getText().toString();
                    factorDouble = Double.parseDouble(factorDisplayString);
                    laneDetector.factor = factorDouble;
                }
                return false;
            }
        });

        checkBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if (isChecked) {
                    laneDetector.ROI = true;
                    Toast.makeText(getApplicationContext(), "SHOWING ROI", Toast.LENGTH_SHORT).show();
                } else {
                    laneDetector.ROI = false;
                    Toast.makeText(getApplicationContext(), "HIDING ROI", Toast.LENGTH_SHORT).show();
                }
            }
        });

        checkBoxLane.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if (isChecked) {
                    Toast.makeText(getApplicationContext(), "LANE ASSIST. ON", Toast.LENGTH_SHORT).show();
                    isAssist = true;
                } else {
                    isAssist = false;
                    Toast.makeText(getApplicationContext(), "LANE ASSIST. OFF", Toast.LENGTH_SHORT).show();
                }
            }
        });

        detect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!isRoutineRunning) {
                    cameraBridgeViewBase.setVisibility(View.VISIBLE);

                    cameraBridgeViewBase.setCvCameraViewListener(new CameraBridgeViewBase.CvCameraViewListener2() {
                        @Override
                        public void onCameraViewStarted(int width, int height) {
                        }

                        @Override
                        public void onCameraViewStopped() {
                        }

                        @Override
                        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
                            Mat currentFrame = inputFrame.rgba();
                            final int mHeight = currentFrame.height();
                            final int mWidth = currentFrame.width();

                            frameCounter++;

                            Mat result;

                            if (frameCounter % 3 == 0) {
                                result = laneDetector.detectLanes(currentFrame, mHeight, mWidth);
                                lastResult = result.clone();
                            } else {
                                if (lastResult != null) {
                                    result = lastResult;
                                } else {
                                    result = currentFrame;
                                }
                            }
                            double distTorightLane = laneDetector.distTorightLane;
                            double distToleftLane = laneDetector.distToLeftLane;
                            double distlanes = laneDetector.betweenLanex;

                            final SpannableString spannableString = TextFormatter.formatText("Right Lane ", "Left Lane   ", "Road          ", distTorightLane, distToleftLane, distlanes, android.graphics.Color.TRANSPARENT, android.graphics.Color.WHITE);
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    textViewTest.setText(spannableString);
                                }
                            });

                            laneout(distToleftLane, distTorightLane, 0.7, isAssist);

                            return result;
                        }
                    });

                    if (OpenCVLoader.initDebug()) {
                        cameraBridgeViewBase.enableView();
                    }

                    isRoutineRunning = true;
                } else {
                    cameraBridgeViewBase.setVisibility(View.GONE);
                    cameraBridgeViewBase.disableView();
                    isRoutineRunning = false;
                }
            }
        });

        detectfile.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 0);
            }
        });

        if (isSoundOn) {
            soundon.setVisibility(View.VISIBLE);
            soundoff.setVisibility(View.GONE);
        } else {
            soundon.setVisibility(View.GONE);
            soundoff.setVisibility(View.VISIBLE);
        }

        soundon.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                isSoundOn = false;
                soundon.setVisibility(View.GONE);
                soundoff.setVisibility(View.VISIBLE);
            }
        });

        soundoff.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                isSoundOn = true;
                soundon.setVisibility(View.VISIBLE);
                soundoff.setVisibility(View.GONE);
            }
        });
    }

    private void laneout(final double distL, double distR, final double delta, final boolean flag) {
        if (flag) {
            if (distL > 0 && distL <= delta) {
                Log.d("LANE_DETECTOR", "Turn Right ->");
            } else if (distR > 0 && distR <= delta) {
                Log.d("LANE_DETECTOR", "<- Turn Left");
            }
        }

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (flag) {
                    if (distL > 0 && distL <= delta) {
                        imageViewAlertL.setVisibility(View.VISIBLE);
                        blinkImageView(context, imageViewAlertL, 3, 500);
                    } else if (distR > 0 && distR <= delta) {
                        imageViewAlertR.setVisibility(View.VISIBLE);
                        blinkImageView(context, imageViewAlertR, 3, 500);
                    }
                }
            }
        });
    }

    private boolean canPlaySound = true;
    private Timer soundTimer;

    private void blinkImageView(final Context context, final ImageView imageView, int numRepetitions, int delayMillis) {
        Animation blinkAnimation = new AlphaAnimation(0.0f, 1.0f);
        blinkAnimation.setDuration(500);
        blinkAnimation.setStartOffset(20);
        blinkAnimation.setRepeatMode(Animation.REVERSE);
        blinkAnimation.setRepeatCount(numRepetitions);

        final MediaPlayer mp = MediaPlayer.create(context, R.raw.alert);

        blinkAnimation.setAnimationListener(new Animation.AnimationListener() {
            @Override
            public void onAnimationStart(Animation animation) {
            }

            @Override
            public void onAnimationEnd(Animation animation) {
                imageView.setVisibility(View.GONE);
                if (mp != null) {
                    mp.stop();
                    mp.release();
                }
            }

            @Override
            public void onAnimationRepeat(Animation animation) {
            }
        });

        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                imageView.startAnimation(blinkAnimation);

                if (canPlaySound && isSoundOn && mp != null) {
                    mp.start();
                    canPlaySound = false;
                    soundTimer = new Timer();
                    soundTimer.schedule(new TimerTask() {
                        @Override
                        public void run() {
                            canPlaySound = true;
                        }
                    }, 1000);
                }
            }
        }, delayMillis);
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(cameraBridgeViewBase);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    void getPermission() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.CAMERA}, 102);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 102 && grantResults.length > 0) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                getPermission();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        videoView.setVisibility(View.VISIBLE);

        if (resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();

            Bitmap bitmap = null;
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
            } catch (IOException e) {
                e.printStackTrace();
            }

            if (bitmap != null) {
                Mat imageMat = new Mat();
                Utils.bitmapToMat(bitmap, imageMat);

                Imgproc.cvtColor(imageMat, imageMat, Imgproc.COLOR_RGBA2RGB);

                Mat result = laneDetector.detectLanes(imageMat, imageMat.height(), imageMat.width());

                Bitmap grayBitmap = Bitmap.createBitmap(result.cols(), result.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(result, grayBitmap);
                videoView.setImageBitmap(grayBitmap);
            }
        }
    }
}
