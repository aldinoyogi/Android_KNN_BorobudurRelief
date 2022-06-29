package com.rana.reliefborobudur;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class KNeighbours extends AppCompatActivity {

    private Button mButton;
    private ImageView mImageView;
    private long adder;
    private long adderDataset;

    private TextView mTextView;
    Mat collectedDatasets;


    @Override
    public void onBackPressed() {
    }


    public void showImageOnImageView(Bitmap currentBitmap){
        mImageView = (ImageView) findViewById(R.id.imageview_captured);
        mImageView.setImageBitmap(currentBitmap);
    }


    public Bitmap convertMatToBitmap(Mat imageCaptured){
        Bitmap mBitmap = Bitmap.createBitmap(imageCaptured.cols(), imageCaptured.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageCaptured, mBitmap, true);
        return mBitmap;
    }


    public Mat convertGetIntentToMat(long adderInput){
        Mat imageCaptured = new Mat(adderInput);
        return imageCaptured;
    }


    public void actionBackButton(){
        mButton = (Button) findViewById(R.id.button_back);
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                getIntent().removeExtra("imageCaptured");
                System.gc();
                finish();
            }
        });
    }


    public Mat resizeImage(Mat inputImage){
        Mat resizedImage = new Mat();
        Size sz = new Size(32,32);
        Imgproc.resize(inputImage, resizedImage, sz);
        return resizedImage;
    }


    public void predictCurrentImageWithKNN(Mat inputImage, Mat listMat) {
//        Mat listMat = new Mat();
        AssetManager assetManager = getAssets();
        List<String> listLabels = new ArrayList<String>();
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
                        listLabels.add(folder.trim());
                    }
                }
            }
            /*====================================================================================*/


            /*================Create Label for KNN OpenCV================*/
            Mat labels = new Mat(1, listLabels.size(), CvType.CV_32F);
            for (int i = 0; i < listLabels.size(); i++){
                labels.put(0, i, i);
            }
            labels.convertTo(labels, CvType.CV_32F);
            /*============================================================*/


            /*=============Training KNN=================*/
            KNearest KNN = KNearest.create();
            KNN.train(listMat, Ml.ROW_SAMPLE, labels);
            /*==========================================*/


            /*====================Start Prediction================*/
            Mat newMat = new Mat();
            float nearestValue;
            String labelPrediction;

            KNN.isClassifier();
            nearestValue = KNN.findNearest(inputImage, 1, newMat);
            labelPrediction = listLabels.get((int)nearestValue);
            /*=====================================================*/


            mTextView = (TextView) findViewById(R.id.textview_prediction);
            mTextView.setText(labelPrediction);
            newMat.release();
            labels.release();
            listMat.release();
            KNN.clear();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public Mat reshapeMat(Mat inputMat){
        Mat reshaped = inputMat.reshape(1, 1);
        reshaped.convertTo(reshaped, CvType.CV_32F);
        return reshaped;
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_kneighbours);
        actionBackButton();

        adder = getIntent().getLongExtra("imageCaptured", 0);
        adderDataset = getIntent().getLongExtra("collectedDataset", 0);

        Mat imageCropped = convertGetIntentToMat(adder);
        Bitmap mBitmap = convertMatToBitmap(imageCropped);
        showImageOnImageView(mBitmap);

        Mat resizedImage = resizeImage(imageCropped);
        Mat reshapedImage = reshapeMat(resizedImage);

        SystemClock.sleep(200);
        predictCurrentImageWithKNN(reshapedImage, convertGetIntentToMat(adderDataset));

        imageCropped.release();
        resizedImage.release();
        reshapedImage.release();
    }
}