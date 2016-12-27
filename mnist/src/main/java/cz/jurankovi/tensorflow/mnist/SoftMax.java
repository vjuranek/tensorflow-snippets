package cz.jurankovi.tensorflow.mnist;

import static org.bytedeco.javacpp.tensorflow.DT_FLOAT;

import java.nio.FloatBuffer;

import org.bytedeco.javacpp.tensorflow.Status;
import org.bytedeco.javacpp.tensorflow.StringTensorPairVector;
import org.bytedeco.javacpp.tensorflow.StringVector;
import org.bytedeco.javacpp.tensorflow.Tensor;
import org.bytedeco.javacpp.tensorflow.TensorShape;
import org.bytedeco.javacpp.tensorflow.TensorVector;

public class SoftMax extends TFModel {

	public SoftMax(String modelPath, int imgPixels, int numClasses, float[] trainLabesl, float[] trainImages, float[] testLabels, float[] testImages) {
	    super(modelPath, imgPixels, numClasses, trainLabesl, trainImages, testLabels, testImages);
	}
	
	public void trainModel() {
	    TensorVector outputs = new TensorVector();
	    Status status = session.Run(new StringTensorPairVector(), new StringVector(), new StringVector("init"), outputs);
	    checkStatus(status);
	    
	    Tensor x = new Tensor(DT_FLOAT, new TensorShape(BATCH_SIZE, imgPixels));
        Tensor y_ = new Tensor(DT_FLOAT, new TensorShape(BATCH_SIZE, numClasses));

        float[] imgBatch = new float[BATCH_SIZE * imgPixels];
        float[] labelBatch = new float[BATCH_SIZE * numClasses];

        int iterations = testImages.length/(BATCH_SIZE * imgPixels);
        for (int i = 0; i < iterations; i++) {
            System.arraycopy(trainImages, i * BATCH_SIZE * imgPixels, imgBatch, 0, BATCH_SIZE * imgPixels);
            System.arraycopy(trainLabels, i * BATCH_SIZE * numClasses, labelBatch, 0, BATCH_SIZE * numClasses);
            FloatBuffer xBuff = x.createBuffer();
            FloatBuffer yBuff = y_.createBuffer();
            
            xBuff.limit(BATCH_SIZE * imgPixels);
            yBuff.limit(BATCH_SIZE * numClasses);
            xBuff.put(imgBatch);
            yBuff.put(labelBatch);
            
            
            status = session.Run(new StringTensorPairVector(new String[] { "x", "y_" }, new Tensor[] { x, y_ }),
                    new StringVector(), new StringVector("train_step"), outputs);
            checkStatus(status);
        }
	}
	
	public void testModel() {
	    Tensor xTest = new Tensor(DT_FLOAT, new TensorShape(BATCH_SIZE, imgPixels));
        Tensor yTest_ = new Tensor(DT_FLOAT, new TensorShape(BATCH_SIZE, numClasses));

        float[] imgTestBatch = new float[BATCH_SIZE * imgPixels];
        float[] labelTestBatch = new float[BATCH_SIZE * numClasses];
        
        System.arraycopy(testImages, 0, imgTestBatch, 0, BATCH_SIZE * imgPixels);
        System.arraycopy(testLabels, 0, labelTestBatch, 0, BATCH_SIZE * numClasses);
        FloatBuffer xTestBuff = xTest.createBuffer();
        FloatBuffer yTestBuff = yTest_.createBuffer();
        
        xTestBuff.limit(BATCH_SIZE * imgPixels);
        yTestBuff.limit(BATCH_SIZE * numClasses);
        xTestBuff.put(imgTestBatch);
        yTestBuff.put(labelTestBatch);
        
        TensorVector outputs = new TensorVector();
        Status status = session.Run(new StringTensorPairVector(new String[] { "x", "y_" }, new Tensor[] { xTest, yTest_ }),
                new StringVector("accuracy"), new StringVector("accuracy"), outputs);
        checkStatus(status);
        
        FloatBuffer output = outputs.get(0).createBuffer();
        for (int i = 0; i < output.limit(); i++) {
            System.out.println("accuracy: " + output.get(i));
        }
	}
	
	public void predictImage(int n) {
	    Tensor x1 = new Tensor(DT_FLOAT, new TensorShape(1, imgPixels));
        float[] imgTest1 = new float[imgPixels];
        
        System.arraycopy(testImages, imgPixels * (n -1), imgTest1, 0, imgPixels);
        FloatBuffer xTest1 = x1.createBuffer();
        
        xTest1.limit(imgPixels);
        xTest1.put(imgTest1);
        
        TensorVector outputs = new TensorVector();
        Status status = session.Run(new StringTensorPairVector(new String[] { "x"}, new Tensor[] { x1 }),
                new StringVector("y"), new StringVector("y"), outputs);
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
