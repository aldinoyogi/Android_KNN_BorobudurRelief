package com.rana.reliefborobudur;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;
import org.w3c.dom.Text;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Log;
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

import androidx.annotation.Nullable;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

public class MainActivity extends Activity implements CvCameraViewListener2 {

    private static final int BOLD = Typeface.BOLD;
    private static String LOG_OpenCV = "OpenCV_Log";

    private Mat matCroppedImage;
    private Mat matCameraImage;
    private Mat matTempImage;

    private int intMatchPrediction;
    private float floatNearestValue;
    private String stringLabelPrediction;

    /*=======================================For KNN=================================*/
    private List<String> listStringLabelImage = new ArrayList<String>();
    private List<Mat> listMatDatasets = new ArrayList<Mat>();

    private List<Mat> listDescriptor_1;
    private List<MatOfKeyPoint> listKeypoint_1;

    private Mat matMaskDetector;

    private ORB featureDetector_1;
    private Mat descriptors_1;
    private MatOfKeyPoint keyPoint_1;

    private ORB featureDetector_2;
    private Mat descriptors_2;
    private MatOfKeyPoint keyPoint_2;

    private BFMatcher bfMatcher;
    private List<MatOfDMatch> listKnnMatches;

    private int intNumberImageDataset;
    /*===============================================================================*/

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

    private TextView mTextViewPrediction;
    private TextView mTextviewAutoCrop;
    private TextView mTextViewDistance;
    private TextView mTextViewDetail;

    private Button mBackButton;
    private ImageView mImageView;
    private ImageView mImageViewDataset;
    private SeekBar seekBarCrop;
    private Switch mSwitchInvert;

    private Bitmap bitmapImageView;
    private Bitmap bitmapImageViewDataset;

    /*========================For AutoCrop=====================*/
    private Mat matTempImage2;
    private MatOfPoint2f approxCurve;
    private MatOfPoint2f contour2f;
    private MatOfPoint points;
    private double doubleDistance;
    private Rect rect;
    /*=========================================================*/

    private DisplayMetrics displayMetrics;
    private float floatScreenWidth;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    try
                    {

                        displayMetrics = getResources().getDisplayMetrics();
                        floatScreenWidth = displayMetrics.widthPixels;

                        matCameraImage = new Mat();
                        matCroppedImage = new Mat();
                        matTempImage = new Mat();
                        matHierarchy = new Mat();
                        matMaskDetector = new Mat();
                        descriptors_2 = new Mat();

                        keyPoint_1 = new MatOfKeyPoint();
                        keyPoint_2 = new MatOfKeyPoint();

                        contours = new ArrayList<MatOfPoint>();

                        listKeypoint_1 = new ArrayList<MatOfKeyPoint>();
                        listDescriptor_1 = new ArrayList<Mat>();
                        listKnnMatches = new ArrayList<MatOfDMatch>();

                        featureDetector_1 = ORB.create();
                        featureDetector_2 = ORB.create();

                        bfMatcher = BFMatcher.create();

                        AssetManager assetManager = getAssets();
                        String[] stringDirPaths = assetManager.list("datasets");

                        for (String stringFilename: stringDirPaths)
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
                                    String stringFullPath = "datasets/"+stringFilename;
                                    inputStream = assetManager.open(stringFullPath);

                                    matTempImage = new Mat();

                                    Bitmap tempBitmap = BitmapFactory.decodeStream(inputStream);
                                    Utils.bitmapToMat(tempBitmap, matTempImage, true);

                                    String ReliefName = stringFilename.split("\\.", 2)[0];
                                    ReliefName = ReliefName.replace(" ", "_");
                                    listStringLabelImage.add(ReliefName);
                                    listMatDatasets.add(matTempImage);


                                    descriptors_1 = new Mat();
                                    featureDetector_1.detectAndCompute(matTempImage, matMaskDetector,
                                            keyPoint_1, descriptors_1);

                                    listKeypoint_1.add(keyPoint_1);
                                    listDescriptor_1.add(descriptors_1);

                                    matMaskDetector.release();
                                } catch (Exception ex)
                                {
                                    ex.printStackTrace();
                                    Log.i(LOG_OpenCV, ex.getMessage());
                                }
                            }
                        }
                        mOpenCvCameraView.enableView();
                        booleanCaptureImage = true;

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

        contours = contours.stream().filter(c -> {
            int minArea = (int)(floatScreenWidth*0.7);
            double area = Imgproc.contourArea(c);
            if (area > minArea) return true;
            return false;
        }).collect(Collectors.toList());

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

        mTextViewPrediction = (TextView) findViewById(R.id.textview_prediction);
        mTextViewDistance = (TextView) findViewById(R.id.textview_distance);
        mTextViewDetail = (TextView) findViewById(R.id.textview_detail);
        mImageViewDataset = (ImageView) findViewById(R.id.imageview_dataset);
        mImageView = (ImageView) findViewById(R.id.imageview_captured);

        mTextViewPrediction.setTypeface(null, BOLD);

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
        mTextviewAutoCrop = (TextView) findViewById(R.id.fab_auto_crop);
        mTextviewAutoCrop.setTextColor(Color.parseColor("#ffffff"));
        mTextviewAutoCrop.setTypeface(null, BOLD);
        mTextviewAutoCrop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                booleanAutoCrop = !booleanAutoCrop;
                if (booleanAutoCrop)
                {
                    mTextviewAutoCrop.setTextColor(Color.parseColor("#ffb600"));
                }
                else
                {
                    mTextviewAutoCrop.setTextColor(Color.parseColor("#ffffff"));
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
                        startPredictFeatures();
                        SystemClock.sleep(300);
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
        mImageView.getLayoutParams().width = (int) floatScreenWidth;
        mImageView.getLayoutParams().height = (int)(floatScreenWidth*0.65);
    }

    private void showImagePreviewDataset(Mat image){
        bitmapImageViewDataset = Bitmap.createBitmap(image.cols(),
                image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(image, bitmapImageViewDataset, true);
        mImageViewDataset.setImageBitmap(bitmapImageViewDataset);
        mImageViewDataset.getLayoutParams().width = (int) floatScreenWidth;
        mImageViewDataset.getLayoutParams().height = (int)(floatScreenWidth*0.65);
    }

    private Mat resizeImage(Mat image){
        int width = image.width();
        int height = image.height();
        int maxValue = 560;
        int maxShape = Math.max(width, height);

        if (maxShape > maxValue)
        {
            int scaleW = maxValue;
            int scaleH = (int)((height*maxValue)/width);
            Imgproc.resize(image, image, new Size(scaleW, scaleH), Imgproc.INTER_AREA);
        }
        else
        {
            int scaleH = maxValue;
            int scaleW = (int)((width*maxValue)/height);
            Imgproc.resize(image, image, new Size(scaleW, scaleH), Imgproc.INTER_AREA);
        }
        return image;
    }

    private void startPredictFeatures(){
        showImagePreview();

        Mat grayMat = new Mat();
        descriptors_2 = new Mat();

        Imgproc.cvtColor(matCroppedImage, grayMat, Imgproc.COLOR_RGBA2GRAY, 1);
        grayMat = resizeImage(grayMat);

        featureDetector_2.detectAndCompute(grayMat, matMaskDetector,
                keyPoint_2, descriptors_2);

        Map<String,Integer> resultMatch = new HashMap<String,Integer>();

        try
        {

            for (int i = 0; i < listDescriptor_1.size(); i++)
            {
                int totalMatch = 0;
                descriptors_1 = new Mat();
                descriptors_1 = listDescriptor_1.get(i);
                bfMatcher.knnMatch(descriptors_1, descriptors_2, listKnnMatches, 3);

                for (int j = 0; j < listKnnMatches.size(); j++)
                {
                    if (listKnnMatches.get(j).rows() > 1)
                    {
                        DMatch[] matches = listKnnMatches.get(j).toArray();
                        if (matches[0].distance < 0.75f * matches[1].distance)
                        {
                            totalMatch = totalMatch + 1;
                        }
                    }
                }
                resultMatch.put(listStringLabelImage.get(i), totalMatch);
            }

            intMatchPrediction = 0;
            stringLabelPrediction = "";
            intNumberImageDataset = 0;

            resultMatch.forEach((k, v) -> {
                if (v > intMatchPrediction)
                {
                    intMatchPrediction = v;
                    stringLabelPrediction = k;
                    intNumberImageDataset = listStringLabelImage.indexOf(stringLabelPrediction);
                }
            });

            if (intMatchPrediction < 11)
            {
                mTextViewPrediction.setText("Unknown");
                mTextViewDistance.setText(String.format("KnnMatch Features: %d", intMatchPrediction));
                mTextViewDetail.setText(getResources().getString(R.string.Relief));
                mImageViewDataset.setImageResource(android.R.color.transparent);
            }
            else
            {
                showImagePreviewDataset(listMatDatasets.get(intNumberImageDataset));
                mTextViewDistance.setText(String.format("KnnMatch Features: %d", intMatchPrediction));
                switch (stringLabelPrediction)
                {
                    case "Relief_1":
                        mTextViewPrediction.setText(getResources().getString(R.string.Relief_1_Title));
                        mTextViewDetail.setText(getResources().getString(R.string.Relief_1));
                        break;
                    case "Relief_2":
                        mTextViewPrediction.setText(getResources().getString(R.string.Relief_2_Title));
                        mTextViewDetail.setText(getResources().getString(R.string.Relief_2));
                        break;
                    case "Relief_3":
                        mTextViewPrediction.setText(getResources().getString(R.string.Relief_3_Title));
                        mTextViewDetail.setText(getResources().getString(R.string.Relief_3));
                        break;
                    case "Relief_4":
                        mTextViewPrediction.setText(getResources().getString(R.string.Relief_4_Title));
                        mTextViewDetail.setText(getResources().getString(R.string.Relief_4));
                        break;
                    case "Relief_41":
                        mTextViewPrediction.setText(getResources().getString(R.string.Relief_41_Title));
                        mTextViewDetail.setText(getResources().getString(R.string.Relief_41));
                        break;
                    default:
                        mTextViewPrediction.setText(stringLabelPrediction);
                        mTextViewDetail.setText(getResources().getString(R.string.Relief));
                }
            }

            listKnnMatches.clear();
        }
        catch (Exception exc) { }
    }

}