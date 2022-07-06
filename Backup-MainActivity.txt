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
import org.opencv.core.MatOfPoint;
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

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
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
import android.widget.CompoundButton;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

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
    private Mat matSelectedDistance;

    private float floatNearestValue;
    private String stringLabelPrediction;

    private List<String> listStringLabelImage = new ArrayList<String>();
    private List<Mat> listMatDatasets = new ArrayList<Mat>();

    private KNearest KNN;
    private int intCameraHeight;
    private int intCameraWidth;
    private int intCropPoint = 0;
    private double doubleImageCrop = 0.2;
    private double doubleImageMaxCrop = 0.4;
    private Boolean booleanPortraitMode = false;
    private Boolean booleanAutoCrop = false;
    private Boolean booleanInvert = false;
    private Boolean booleanCaptureImage = false;
    private Size sizeKnnPredict = new Size(32, 32);
    private List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    private Mat matHierarchy;

    private InputStream inputStream = null;
    private CameraBridgeViewBase mOpenCvCameraView;

    private FloatingActionButton captureButton;
    private FloatingActionButton portraitButton;
    private FloatingActionButton landscapeButton;

    private FrameLayout layoutMain;
    private FrameLayout layoutView;

    private TextView mTextView;
    private TextView mAutoCrop;
    private TextView mDistance;

    private Button mBackButton;
    private ImageView mImageView;
    private SeekBar seekBarCrop;
    private Switch mSwitchInvert;

    private Bitmap bitmapImageView;

    /*========================For AutoCrop=====================*/
    private Mat matTempImage2;
    private MatOfPoint2f approxCurve;
    private MatOfPoint2f contour2f;
    private MatOfPoint points;
    private double doubleDistance;
    private Rect rect;
    /*=========================================================*/


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
                        matSelectedDistance = new Mat();
                        matHierarchy = new Mat();
                        contours = new ArrayList<MatOfPoint>();

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
                                        listMatDatasets.add(matTempImage.reshape(1,1));

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
                        booleanCaptureImage = true;

                        KNN = KNearest.create();
                        KNN.train(matDatasetImage, Ml.ROW_SAMPLE, matLabelIntImage);

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
        //Log.i(LOG_OpenCV, String.format("Height: %d, Width: %d", height, width));
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        matCameraImage = inputFrame.rgba();
        intCropPoint = (int)(intCameraHeight*doubleImageCrop);

        if (booleanAutoCrop)
        {
            try { findContoursOfImage(); }
            catch (Exception exc) {};
        }
        else
        {
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
        }


        return matCameraImage;
    }


    private void findContoursOfImage(){

        matTempImage2 = matCameraImage.clone();
        Imgproc.cvtColor(matCameraImage, matTempImage, Imgproc.COLOR_RGBA2GRAY);
        if (booleanInvert)
        {
            Imgproc.threshold(matTempImage, matTempImage, 155,200, Imgproc.THRESH_BINARY);
        }
        else
        {
            Imgproc.threshold(matTempImage, matTempImage, 60,155, Imgproc.THRESH_BINARY_INV);
        }
        Imgproc.findContours(matTempImage, contours, matHierarchy,
                Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
        matHierarchy.release();

        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++)
        {
            approxCurve = new MatOfPoint2f();
            contour2f = new MatOfPoint2f(contours.get(contourIdx).toArray());

            doubleDistance = Imgproc.arcLength(contour2f, true)*0.01;
            Imgproc.approxPolyDP(contour2f, approxCurve, doubleDistance, true);

            points = new MatOfPoint(approxCurve.toArray());
            rect = Imgproc.boundingRect(points);

            double doubleHeight = rect.height;
            double doubleWidth = rect.width;

            if (doubleWidth > Math.min(intCameraWidth, intCameraHeight)/5
                && doubleHeight > Math.min(intCameraWidth, intCameraHeight)/5
                && doubleHeight < intCameraHeight-20
                && doubleWidth < intCameraWidth-20)
            {
                int threshold = 5;
                double currentX = rect.x + threshold;
                double currentY = rect.y + threshold;
                double distanceX = doubleWidth - threshold*2;
                double distanceY = doubleHeight - threshold*2;
                Imgproc.rectangle(matCameraImage, new Point(currentX, currentY),
                        new Point(currentX + distanceX, currentY + distanceY),
                        new Scalar(255,223,0), 3);
                try
                {
                    Rect roi = new Rect((int)currentX,
                                (int)currentY,
                                (int) distanceX,
                                (int) distanceY);
                    matCroppedImage = new Mat(matTempImage2, roi);
                    Core.rotate(matCroppedImage, matCroppedImage, 0);
                } catch (Exception exc) { Log.i(LOG_OpenCV, exc.getMessage()); }
                matHierarchy.release();
                contours.clear();
                break;
            }
            else
            {
                matCroppedImage = null;
            }
        }
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
        mDistance = (TextView) findViewById(R.id.textview_distance);
        mImageView = (ImageView) findViewById(R.id.imageview_captured);

        mTextView.setTypeface(null, BOLD);

        onClickAutoCropImage();
        onChangeSwitchInvert();
        onChangeSeekbarForCropping();
        onChangeOrientationCropping();
        onClickCaptureImage();
        onClickBackViewCamera();

    }


    private void onChangeSwitchInvert(){
        mSwitchInvert = (Switch) findViewById(R.id.switch_invert);
        mSwitchInvert.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                booleanInvert = !booleanInvert;
            }
        });
    }

    private void onClickAutoCropImage(){
        mAutoCrop = (TextView) findViewById(R.id.fab_auto_crop);
        mAutoCrop.setTextColor(Color.parseColor("#ffffff"));
        mAutoCrop.setTypeface(null, BOLD);
        mAutoCrop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                booleanAutoCrop = !booleanAutoCrop;
                if (booleanAutoCrop)
                {
                    mAutoCrop.setTextColor(Color.parseColor("#ffb600"));
                }
                else
                {
                    mAutoCrop.setTextColor(Color.parseColor("#ffffff"));
                }
            }
        });
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
                if (matCroppedImage != null && booleanCaptureImage){
                    layoutMain.setVisibility(View.GONE);
                    layoutView.setVisibility(View.VISIBLE);
                    try
                    {
                        SystemClock.sleep(300);
                        predictCurrentImage();
                    }
                    catch (Exception exc) { }
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

    private void showImagePreview(){
        bitmapImageView = Bitmap.createBitmap(matCroppedImage.cols(),
                matCroppedImage.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matCroppedImage, bitmapImageView, true);
        mImageView.setImageBitmap(bitmapImageView);
    }

    private double findDistance(int intNearestValue){
        double totalEuc = 0;
        matSelectedDistance = listMatDatasets.get(intNearestValue);
        for (int row = 0; row < matSelectedDistance.rows(); row++)
        {
            for (int col = 0; col < matSelectedDistance.cols(); col++)
            {
                double[] pixelSelectedImage = matSelectedDistance.get(row, col);
                double[] pixelTempImage = matTempImage.get(row, col);

                for (int colPix = 0; colPix < pixelSelectedImage.length; colPix++)
                {
                    if (colPix != 3)
                    {
                        totalEuc = totalEuc + Math.pow(pixelTempImage[colPix]
                                - pixelSelectedImage[colPix], 2);
                    }
                }
            }
        }

        totalEuc = Math.sqrt(Math.abs(totalEuc));
        //Log.i(LOG_OpenCV, String.format("Distance: %f", totalEuc));
        matSelectedDistance.release();

        return totalEuc;
    }


    private void predictCurrentImage(){

        Imgproc.resize(matCroppedImage, matTempImage, sizeKnnPredict);

        matTempImage = matTempImage.reshape(1,1);
        matTempImage.convertTo(matTempImage, CvType.CV_32F);

        floatNearestValue = KNN.findNearest(matTempImage,1,matImagePredicted);
        stringLabelPrediction = listStringLabelImage.get((int)floatNearestValue);
        showImagePreview();

        double doubleDistance = findDistance((int)floatNearestValue);
        if (doubleDistance < 2800)
        {
            mTextView.setText(stringLabelPrediction);
        }
        else
        {
            mTextView.setText("Unknown");
        }
        mDistance.setText(String.format("Distance: %.2f", doubleDistance));
        mDistance.setTypeface(null, Typeface.ITALIC);
    }


}