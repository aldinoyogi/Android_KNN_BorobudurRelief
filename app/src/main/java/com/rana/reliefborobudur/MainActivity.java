package com.rana.reliefborobudur;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.annotation.Nullable;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

public class MainActivity extends Activity implements CvCameraViewListener2 {

    private static final int BOLD = Typeface.BOLD;
    private static String LOG_OpenCV = "OpenCV_Log";

    private Mat matCroppedImage;
    private Mat matCameraImage;
    private Mat matTempImage;
    private Mat matDatasetImage;
    private Mat matLabelIntImage;
    private Mat matImagePredicted;

    private float floatNearestValue;
    private String stringLabelPrediction;

    private List<String> listStringLabelImage = new ArrayList<String>();

    private int intCameraHeight;
    private int intCameraWidth;
    private int intCropPoint = 0;
    private double doubleImageCrop = 0.2;
    private double doubleImageMaxCrop = 0.4;
    private Boolean booleanPortraitMode = false;
    private Size sizeKnnPredict = new Size(32, 32);

    private InputStream inputStream = null;
    private CameraBridgeViewBase mOpenCvCameraView;

    private FloatingActionButton captureButton;
    private FloatingActionButton portraitButton;
    private FloatingActionButton landscapeButton;

    private FrameLayout layoutMain;
    private FrameLayout layoutView;

    private TextView mTextView;
    private Button mBackButton;
    private ImageView mImageView;
    private SeekBar seekBarCrop;

    private Bitmap bitmapImageView;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    try
                    {
                        matCameraImage = new Mat();
                        matCroppedImage = new Mat();
                        matTempImage = new Mat();
                        matDatasetImage = new Mat();
                        matImagePredicted = new Mat();

                        AssetManager assetManager = getAssets();
                        String[] stringDirPaths = assetManager.list("datasets");

                        for (String stringFolderName: stringDirPaths)
                        {
                            String[] stringFilenames = assetManager.list("datasets/"+stringFolderName);
                            for (String stringFilename: stringFilenames)
                            {
                                if
                                (
                                    stringFilename.endsWith(".jpg")||
                                    stringFilename.endsWith(".png") ||
                                    stringFilename.endsWith(".jpeg")
                                )
                                {
                                    try
                                    {
                                        String stringFullPath = "datasets/"+stringFolderName+"/"+stringFilename;
                                        inputStream = assetManager.open(stringFullPath);

                                        Bitmap tempBitmap = BitmapFactory.decodeStream(inputStream);
                                        Utils.bitmapToMat(tempBitmap, matTempImage, true);

                                        matTempImage = matTempImage.reshape(1,1);
                                        matTempImage.convertTo(matTempImage, CvType.CV_32F);

                                        matDatasetImage.push_back(matTempImage.reshape(1,1));
                                        listStringLabelImage.add(stringFolderName);

                                        matTempImage.release();

                                    } catch (Exception ex)
                                    {
                                        ex.printStackTrace();
                                    }
                                }
                            }
                        }
                        matLabelIntImage = new Mat(1, listStringLabelImage.size(), CvType.CV_32F);
                        for (int i = 0; i < listStringLabelImage.size(); i++){
                            matLabelIntImage.put(0, i, i);
                        }
                        matLabelIntImage.convertTo(matLabelIntImage, CvType.CV_32F);
                        mOpenCvCameraView.enableView();

                    } catch (Exception ex)
                    {
                        ex.printStackTrace();
                    }
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        intCameraHeight = height;
        intCameraWidth = width;
        Log.i(LOG_OpenCV, String.format("Height: %d, Width: %d", height, width));
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        SystemClock.sleep(10);
        matCameraImage = inputFrame.rgba();
        intCropPoint = (int)(intCameraHeight*doubleImageCrop);

        if (booleanPortraitMode)
        {
            Imgproc.rectangle(matCameraImage, new Point(0, intCropPoint),
                    new Point(intCameraWidth, intCameraHeight - intCropPoint),
                    new Scalar(255,223,0), 2);
            Rect roi = new Rect(0, intCropPoint, intCameraWidth,
                    intCameraHeight - (intCropPoint*2));
            matCroppedImage = new Mat(matCameraImage, roi);
        }
        else
        {
            Imgproc.rectangle(matCameraImage, new Point(intCropPoint, 0),
                    new Point(intCameraWidth - intCropPoint, intCameraHeight),
                    new Scalar(255,223,0), 2);
            Rect roi = new Rect(intCropPoint, 0, intCameraWidth - (intCropPoint*2),
                    intCameraHeight);
            matCroppedImage = new Mat(matCameraImage, roi);
        }
        Core.rotate(matCroppedImage, matCroppedImage, 0);
        return matCameraImage;
    }


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_stream_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mTextView = (TextView) findViewById(R.id.textview_prediction);
        mImageView = (ImageView) findViewById(R.id.imageview_captured);

        mTextView.setTypeface(null, BOLD);

        onChangeSeekbarForCropping();
        onChangeOrientationCropping();
        onClickCaptureImage();
        onClickBackViewCamera();

    }

    private void onChangeSeekbarForCropping(){
        seekBarCrop = (SeekBar) findViewById(R.id.seekbar_crop);
        seekBarCrop.setProgress(50);
        seekBarCrop.setMax(95);
        seekBarCrop.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                doubleImageCrop = doubleImageMaxCrop-((doubleImageMaxCrop/100)*progress);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
    }

    private void onChangeOrientationCropping(){
        layoutMain = (FrameLayout) findViewById(R.id.layout_main);
        layoutView = (FrameLayout) findViewById(R.id.layout_view);

        landscapeButton = (FloatingActionButton) findViewById(R.id.fab_landscape);
        portraitButton = (FloatingActionButton) findViewById(R.id.fab_portrait);

        landscapeButton.setVisibility(View.GONE);
        landscapeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                booleanPortraitMode = false;
                landscapeButton.setVisibility(View.GONE);
                portraitButton.setVisibility(View.VISIBLE);
            }
        });

        portraitButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                booleanPortraitMode = true;
                landscapeButton.setVisibility(View.VISIBLE);
                portraitButton.setVisibility(View.GONE);
            }
        });

    }

    private void onClickCaptureImage(){
        captureButton = (FloatingActionButton) findViewById(R.id.button_capture);
        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (matCroppedImage != null){
                    layoutMain.setVisibility(View.GONE);
                    layoutView.setVisibility(View.VISIBLE);
                    predictCurrentImage();
                }
            }
        });
    }

    private void onClickBackViewCamera(){
        mBackButton = (Button) findViewById(R.id.button_back);
        mBackButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                layoutMain.setVisibility(View.VISIBLE);
                layoutView.setVisibility(View.GONE);
                mImageView.setImageResource(android.R.color.transparent);
            }
        });
    }

    private void predictCurrentImage(){

        showImagePreview();
        Imgproc.resize(matCroppedImage, matTempImage, sizeKnnPredict);

        matTempImage = matTempImage.reshape(1,1);
        matTempImage.convertTo(matTempImage, CvType.CV_32F);

        KNearest KNN = KNearest.create();
        KNN.train(matDatasetImage, Ml.ROW_SAMPLE, matLabelIntImage);

        floatNearestValue = KNN.findNearest(matTempImage,5,matImagePredicted);
        stringLabelPrediction = listStringLabelImage.get((int)floatNearestValue);

        mTextView.setText(stringLabelPrediction);
    }

    private void showImagePreview(){
        bitmapImageView = Bitmap.createBitmap(matCroppedImage.cols(),
                matCroppedImage.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matCroppedImage, bitmapImageView, true);
        mImageView.setImageBitmap(bitmapImageView);
    }
}