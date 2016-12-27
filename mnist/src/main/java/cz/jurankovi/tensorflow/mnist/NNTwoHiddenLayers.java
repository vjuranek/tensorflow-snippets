package cz.jurankovi.tensorflow.mnist;

import static org.bytedeco.javacpp.tensorflow.DT_FLOAT;
import static org.bytedeco.javacpp.tensorflow.DT_INT32;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import org.bytedeco.javacpp.tensorflow.Status;
import org.bytedeco.javacpp.tensorflow.StringTensorPairVector;
import org.bytedeco.javacpp.tensorflow.StringVector;
import org.bytedeco.javacpp.tensorflow.Tensor;
import org.bytedeco.javacpp.tensorflow.TensorShape;
import org.bytedeco.javacpp.tensorflow.TensorVector;

public class NNTwoHiddenLayers extends TFModel {

    private final int[] trainLabelsInt;
    private final int[] testLabelsInt;
    
    public NNTwoHiddenLayers(String modelPath, int imgPixels, float[] trainLabesl, float[] trainImages, float[] testLabels, float[] testImages) {
        super(modelPath, imgPixels, 0, trainLabesl, trainImages, testLabels, testImages);
        //TODO fix on parent
        trainLabelsInt = new int[trainLabels.length];
        for (int i = 0; i < trainLabels.length; i++) {
            trainLabelsInt[i] = (int)trainLabels[i];
        }
        testLabelsInt = new int[testLabels.length];
        for (int i = 0; i < testLabels.length; i++) {
            testLabelsInt[i] = (int)testLabels[i];
        }
    }
    
    public void trainModel() {
        TensorVector outputs = new TensorVector();
        Status status = session.Run(new StringTensorPairVector(), new StringVector(), new StringVector("init"), outputs);
        checkStatus(status);
        
        
        Tensor images = new Tensor(DT_FLOAT, new TensorShape(BATCH_SIZE, imgPixels));
        TensorShape labelsShape = new TensorShape();  //TensorShape(BATCH_SIZE) return wrong number of elements (just one) for some reason
        labelsShape.AddDim(BATCH_SIZE);
        Tensor labels = new Tensor(DT_INT32, labelsShape);
        
        float[] imgTrainBatch = new float[BATCH_SIZE * imgPixels];
        int[] labelTrainBatch = new int[BATCH_SIZE];
        
        int iterations = trainImages.length/(BATCH_SIZE * imgPixels);
        for (int i = 0; i < iterations; i++) {
            System.arraycopy(trainImages, i * BATCH_SIZE * imgPixels, imgTrainBatch, 0, BATCH_SIZE * imgPixels);
            System.arraycopy(trainLabelsInt, i * BATCH_SIZE, labelTrainBatch, 0, BATCH_SIZE);
            FloatBuffer imagesBuff = images.createBuffer();
            IntBuffer labelsBuff = labels.createBuffer();
            
            imagesBuff.limit(BATCH_SIZE * imgPixels);
            labelsBuff.limit(BATCH_SIZE);
            imagesBuff.put(imgTrainBatch);
            labelsBuff.put(labelTrainBatch);
            
            status = session.Run(new StringTensorPairVector(new String[] { "images", "labels" }, new Tensor[] { images, labels }),
                    new StringVector(), new StringVector("xentropy_mean", "train"), outputs);
            checkStatus(status);
        }
    }
    
    public void testModel() {
        Tensor images = new Tensor(DT_FLOAT, new TensorShape(BATCH_SIZE, imgPixels));
        TensorShape labelsShape = new TensorShape();  //TensorShape(BATCH_SIZE) return wrong number of elements (just one) for some reason
        labelsShape.AddDim(BATCH_SIZE);
        Tensor labels = new Tensor(DT_INT32, labelsShape);
        
        FloatBuffer imagesBuff = images.createBuffer();
        IntBuffer labelsBuff = labels.createBuffer();
        float[] imgTestBatch = new float[BATCH_SIZE * imgPixels];
        int[] labelTestBatch = new int[BATCH_SIZE];
        System.arraycopy(testImages, 0, imgTestBatch, 0, BATCH_SIZE * imgPixels);
        System.arraycopy(testLabelsInt, 0, labelTestBatch, 0, BATCH_SIZE);
        
        imagesBuff.limit(BATCH_SIZE * imgPixels);
        labelsBuff.limit(BATCH_SIZE);
        imagesBuff.put(imgTestBatch);
        labelsBuff.put(labelTestBatch);
        
        TensorVector outputs = new TensorVector();
        Status status = session.Run(new StringTensorPairVector(new String[] { "images", "labels" }, new Tensor[] { images, labels }),
                new StringVector("eval_correct"), new StringVector("eval_correct"), outputs);
        checkStatus(status);

        IntBuffer output = outputs.get(0).createBuffer();
        for (int i = 0; i < output.limit(); i++) {
            System.out.println("correct entries: " + output.get(i));
        }

    }
    
    public void predictImage(int n) {
        Tensor img = new Tensor(DT_FLOAT, new TensorShape(1, imgPixels));
        float[] imgTest1 = new float[imgPixels];
        
        System.arraycopy(testImages, imgPixels * (n -1), imgTest1, 0, imgPixels);
        FloatBuffer imgBuff = img.createBuffer();
        
        imgBuff.limit(imgPixels);
        imgBuff.put(imgTest1);
        
        TensorVector outputs = new TensorVector();
        Status status = session.Run(new StringTensorPairVector(new String[] { "images"}, new Tensor[] { img }),
                new StringVector("softmax_linear/logits"), new StringVector("softmax_linear/logits"), outputs);
        checkStatus(status);
        
        FloatBuffer output = outputs.get(0).createBuffer();
        int maxPos = 0;
        float maxY = Float.MIN_VALUE;
        for (int i = 0; i < output.limit(); i++) {
            float yi = output.get(i);
            if (yi > maxY) {
                maxY = yi;
                maxPos = i;
            }
        }
        System.out.printf("%d test image could be %d", n, maxPos);
    }

}
