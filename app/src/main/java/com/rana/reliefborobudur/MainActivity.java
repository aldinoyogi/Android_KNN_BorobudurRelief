package com.rana.reliefborobudur;

import static org.opencv.android.LoaderCallbackInterface.SUCCESS;
import static org.opencv.core.Core.flip;


import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity {

    private static String LOGTAG = "OpenCV_Log";
    private CameraBridgeViewBase cameraBridgeViewBase;
    private FloatingActionButton captureButton;
    private FloatingActionButton portraitButton;
    private FloatingActionButton landscapeButton;
    private FrameLayout layoutMain;
    private FrameLayout layoutView;
    private TextView mTextView;
    private Button mBackButton;
    private ImageView mImageView;
    private SeekBar seekBarCrop;

    Boolean portraitMode = false;
    int imageHeight = 0;
    int imageWidth = 0;
    double imageCropHeight = 0.20;

    List<Mat> collectedDataset = new ArrayList<Mat>();
    List<Mat> capturedImage = new ArrayList<Mat>();
    List<String> listLabelStrings = new ArrayList<String>();
    List<Mat> listLabelIntegers = new ArrayList<Mat>();


    private Mat collectingDatasets(){
        collectedDataset.clear();
        listLabelIntegers.clear();
        listLabelStrings.clear();


        Mat listMat = new Mat();
        AssetManager assetManager = getAssets();
        Mat temporaryImage = new Mat();
        InputStream istr = null;

        /*====================================Collecting Dataset==================================*/
        try {

            String[] dirPaths = assetManager.list("datasets");

            for (String folder:dirPaths){

                String[] fileNames = assetManager.list("datasets/"+folder);

                for (String fileName:fileNames){
                    if(fileName.endsWith(".jpg") || fileName.endsWith(".png")
                            || fileName.endsWith(".jpeg")){

                        try {
                            istr = assetManager.open("datasets/"+folder+"/"+fileName);
                            Bitmap bitmap = BitmapFactory.decodeStream(istr);
                            Utils.bitmapToMat(bitmap, temporaryImage, true);
                            temporaryImage = temporaryImage.reshape(1, 1);
                            temporaryImage.convertTo(temporaryImage, CvType.CV_32F);
                            listMat.push_back(temporaryImage.reshape(1,1));
                            listLabelStrings.add(folder.trim());
                            temporaryImage.release();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                    }
                }
            }
            Mat labels = new Mat(1, listLabelStrings.size(), CvType.CV_32F);
            for (int i = 0; i < listLabelStrings.size(); i++){
                labels.put(0, i, i);
            }
            labels.convertTo(labels, CvType.CV_32F);
            listLabelIntegers.add(labels);

            return listMat;
        }  catch (IOException e) {
            Log.i(LOGTAG, e.getMessage());
            e.printStackTrace();
        }
        SystemClock.sleep(500);
        return null;
    }

    private BaseLoaderCallback loaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:{
                    cameraBridgeViewBase.enableView();
                }
                default:{
                    super.onManagerConnected(status);
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        layoutMain = (FrameLayout) findViewById(R.id.layout_main);
        layoutView = (FrameLayout) findViewById(R.id.layout_view);

        mTextView = (TextView) findViewById(R.id.textview_prediction);
        mImageView = (ImageView) findViewById(R.id.imageview_captured);
        seekBarCrop = (SeekBar) findViewById(R.id.seekbar_crop);

        seekBarCrop.setProgress(50);

        seekBarCrop.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                Log.i(LOGTAG, String.format("%d", progress));
                imageCropHeight = 0.004*progress;
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });

        mBackButton = (Button) findViewById(R.id.button_back);

        mBackButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                layoutMain.setVisibility(View.VISIBLE);
                layoutView.setVisibility(View.GONE);
                mImageView.setImageResource(android.R.color.transparent);
            }
        });

        layoutView.setVisibility(View.INVISIBLE);

        landscapeButton = (FloatingActionButton) findViewById(R.id.fab_landscape);
        portraitButton = (FloatingActionButton) findViewById(R.id.fab_portrait);

        landscapeButton.setVisibility(View.GONE);
        landscapeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                portraitMode = false;
                landscapeButton.setVisibility(View.GONE);
                portraitButton.setVisibility(View.VISIBLE);
            }
        });

        portraitButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                portraitMode = true;
                landscapeButton.setVisibility(View.VISIBLE);
                portraitButton.setVisibility(View.GONE);
            }
        });

        cameraBridgeViewBase = (CameraBridgeViewBase) findViewById(R.id.camera_stream_view);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(cameraViewListener);

        captureButton = (FloatingActionButton) findViewById(R.id.button_capture);
        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (capturedImage.size() > 0){
                    layoutMain.setVisibility(View.GONE);
                    layoutView.setVisibility(View.VISIBLE);
                    startPrediction();

                }
            }
        });
    }

    @Override
    protected List<?extends CameraBridgeViewBase> getCameraViewList(){
        return Collections.singletonList(cameraBridgeViewBase);
    }

    private CameraBridgeViewBase.CvCameraViewListener cameraViewListener =
            new CameraBridgeViewBase.CvCameraViewListener(){
        @Override
        public void onCameraViewStarted(int width, int height) {
            imageHeight = height;
            imageWidth = width;
            collectedDataset.add(collectingDatasets());
        }

        @Override
        public void onCameraViewStopped() {

        }


        @Override
        public Mat onCameraFrame(Mat inputFrame) {
            Core.rotate(inputFrame, inputFrame, 0);
            int topPoint = (int)(imageHeight*imageCropHeight);
            SystemClock.sleep(15);
            capturedImage.clear();
            if(portraitMode){
                Imgproc.rectangle(inputFrame, new Point(topPoint, 0),
                        new Point(imageWidth-topPoint, imageHeight),
                        new Scalar(255,223,0), 2);
                Rect roi = new Rect(topPoint, 0, imageWidth - (topPoint*2), imageHeight);
                Mat newMat = new Mat(inputFrame, roi);
                capturedImage.add(newMat);
            } else {
                Imgproc.rectangle(inputFrame, new Point(0, topPoint),
                        new Point(imageWidth, imageHeight-topPoint),
                        new Scalar(255,223,0), 2);
                Rect roi = new Rect(0, topPoint, imageWidth, imageHeight - (topPoint*2));
                Mat newMat = new Mat(inputFrame, roi);
                capturedImage.add(newMat);
            }
            System.gc();
            return inputFrame;
        }

    };

    @Override
    public void onPause() {
        super.onPause();
        if (cameraBridgeViewBase != null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()){
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, loaderCallback);
        } else {
            loaderCallback.onManagerConnected(SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase != null){
            cameraBridgeViewBase.disableView();
        }
    }

    /*============================================================================================*/
    /*============================================================================================*/
    /*============================================================================================*/

    public void startPrediction(){
        Mat imageCropped = capturedImage.get(0);
        Bitmap mBitmap = convertMatToBitmap(imageCropped);
        showImageOnImageView(mBitmap);

        Mat resizedImage = resizeImage(imageCropped);
        Mat reshapedImage = reshapeMat(resizedImage);
        predictCurrentImageWithKNN(reshapedImage, collectedDataset.get(0));
    }

    public void showImageOnImageView(Bitmap currentBitmap){
        mImageView.setImageBitmap(currentBitmap);
    }


    public Bitmap convertMatToBitmap(Mat imageCaptured){
        Bitmap mBitmap = Bitmap.createBitmap(imageCaptured.cols(), imageCaptured.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageCaptured, mBitmap, true);
        return mBitmap;
    }



    public Mat resizeImage(Mat inputImage){
        Mat resizedImage = new Mat();
        Size sz = new Size(32,32);
        Imgproc.resize(inputImage, resizedImage, sz);
        return resizedImage;
    }


    public Mat reshapeMat(Mat inputMat){
        Mat reshaped = inputMat.reshape(1, 1);
        reshaped.convertTo(reshaped, CvType.CV_32F);
        return reshaped;
    }

    public void predictCurrentImageWithKNN(Mat inputImage, Mat listMat) {
        /*=============Training KNN=================*/
        KNearest KNN = KNearest.create();
        KNN.train(collectedDataset.get(0), Ml.ROW_SAMPLE, listLabelIntegers.get(0));
        /*==========================================*/


        /*====================Start Prediction================*/
        Mat newMat = new Mat();
        float nearestValue;
        String labelPrediction;

        KNN.isClassifier();
        nearestValue = KNN.findNearest(inputImage, 1, newMat);
        labelPrediction = listLabelStrings.get((int)nearestValue);
        /*=====================================================*/

        mTextView.setText(labelPrediction);
        newMat.release();
        KNN.clear();
        System.gc();
    }



}