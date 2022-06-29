package com.rana.reliefborobudur;

import static org.opencv.android.LoaderCallbackInterface.SUCCESS;
import static org.opencv.core.Core.flip;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import java.util.Collections;
import java.util.List;

public class CameraStreamOpenCV extends CameraActivity {

    private static String LOGTAG = "OpenCV_Log";
    private CameraBridgeViewBase cameraBridgeViewBase;

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
        setContentView(R.layout.activity_camera_stream_open_cv);
        cameraBridgeViewBase = (CameraBridgeViewBase) findViewById(R.id.camera_stream_view);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(cameraViewListener2);
    }

    @Override
    protected List<?extends CameraBridgeViewBase> getCameraViewList(){
        return Collections.singletonList(cameraBridgeViewBase);
    }

    public Mat rot90(Mat matImage, int rotflag){
        Mat rotated = new Mat();
        if (rotflag == 1){
            rotated = matImage.t();
            flip(rotated, rotated, 1);
        } else if (rotflag == 2) {
            rotated = matImage.t();
            flip(rotated, rotated,0);
        } else if (rotflag ==3){
            flip(matImage, rotated,-1);
        } else if (rotflag != 0){
            Log.e(LOGTAG, "Unknown rotation flag("+rotflag + ")");
        }
        return rotated;
    }

    private CameraBridgeViewBase.CvCameraViewListener2 cameraViewListener2 = new CameraBridgeViewBase.CvCameraViewListener2(){
        @Override
        public void onCameraViewStarted(int width, int height) {
            Toast.makeText(CameraStreamOpenCV.this, String.format("Width: %d, Height: %d", width, height), Toast.LENGTH_LONG).show();
        }

        @Override
        public void onCameraViewStopped() {

        }

        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
            return rot90(inputFrame.rgba(), 1);
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
            Log.d(LOGTAG, "OpenCV not found");
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