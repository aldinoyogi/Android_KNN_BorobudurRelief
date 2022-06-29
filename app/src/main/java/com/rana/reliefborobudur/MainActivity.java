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
import org.opencv.imgproc.Imgproc;

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

    Boolean portraitMode = false;
    int imageHeight = 0;
    int imageWidth = 0;
    double imageCropHeight = 0.19; // Just 0.0 - 0.45

    Mat collectedDataset;
    Mat capturedImage;
    Boolean savePicture;
    Boolean collected = false;

    private Mat collectingDatasets(){
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
                            temporaryImage.release();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                    }
                }
            }
            SystemClock.sleep(500);
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
                if (capturedImage != null){
                    long capt = capturedImage.getNativeObjAddr();
                    long coll = collectedDataset.getNativeObjAddr();
                    Intent kNeighboursActivity = new Intent(MainActivity.this, KNeighbours.class);
                    kNeighboursActivity.putExtra("imageCaptured", capt);
                    kNeighboursActivity.putExtra("collectedDataset", coll);
                    startActivity(kNeighboursActivity);
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
            collectedDataset = collectingDatasets();
        }

        @Override
        public void onCameraViewStopped() {

        }


        @Override
        public Mat onCameraFrame(Mat inputFrame) {
            Core.rotate(inputFrame, inputFrame, 0);
            int topPoint = (int)(imageHeight*imageCropHeight);
            SystemClock.sleep(30);
            if(portraitMode){
                Imgproc.rectangle(inputFrame, new Point(topPoint, 0),
                        new Point(imageWidth-topPoint, imageHeight),
                        new Scalar(255,223,0), 2);
                Rect roi = new Rect(topPoint, 0, imageWidth - (topPoint*2), imageHeight);
                capturedImage = new Mat(inputFrame, roi);
            } else {
                Imgproc.rectangle(inputFrame, new Point(0, topPoint),
                        new Point(imageWidth, imageHeight-topPoint),
                        new Scalar(255,223,0), 2);
                Rect roi = new Rect(0, topPoint, imageWidth, imageHeight - (topPoint*2));
                capturedImage = new Mat(inputFrame, roi);
            }
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
}