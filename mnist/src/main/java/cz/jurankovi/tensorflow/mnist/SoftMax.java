package cz.jurankovi.tensorflow.mnist;

import static org.bytedeco.javacpp.tensorflow.DT_FLOAT;
import static org.bytedeco.javacpp.tensorflow.DT_STRING;
import static org.bytedeco.javacpp.tensorflow.InitMain;
import static org.bytedeco.javacpp.tensorflow.ReadBinaryProto;

import java.nio.FloatBuffer;

import org.bytedeco.javacpp.tensorflow.Env;
import org.bytedeco.javacpp.tensorflow.GraphDef;
import org.bytedeco.javacpp.tensorflow.Session;
import org.bytedeco.javacpp.tensorflow.SessionOptions;
import org.bytedeco.javacpp.tensorflow.Status;
import org.bytedeco.javacpp.tensorflow.StringTensorPairVector;
import org.bytedeco.javacpp.tensorflow.StringVector;
import org.bytedeco.javacpp.tensorflow.Tensor;
import org.bytedeco.javacpp.tensorflow.TensorShape;
import org.bytedeco.javacpp.tensorflow.TensorVector;
import org.bytedeco.javacpp.helper.tensorflow.StringArray;

public class SoftMax {

	public static final int BATCH_SIZE = 10000;
	private final String modelPath;
	private final float[] trainLabels;
	private final float[] trainImages;
	private final float[] testLabels;
    private final float[] testImages;
    private final int imgPixels;
    private final int numClasses;
    
    private final Session session;

	public SoftMax(String modelPath, int imgPixels, int numClasses, float[] trainLabesl, float[] trainImages, float[] testLabels, float[] testImages) {
		this.modelPath = modelPath;
		this.trainLabels = trainLabesl;
		this.trainImages = trainImages;
		this.testLabels = testLabels;
		this.testImages = testImages;
		this.imgPixels = imgPixels;
		this.numClasses = numClasses;
		
		InitMain("softmax", (int[]) null, null);
        GraphDef graph = new GraphDef();
        ReadBinaryProto(Env.Default(), modelPath, graph);
        SessionOptions options = new SessionOptions();
        this.session = new Session(options);
        Status status = session.Create(graph);
        checkStatus(status);
	}
	
	public void loadCheckpoint(String checkpointPath) {
	    Tensor cpPath = new Tensor(DT_STRING, new TensorShape(1));
        StringArray strArray = cpPath.createStringArray();
        strArray.position(0).put(checkpointPath); 
        Status status = session.Run(new StringTensorPairVector(new String[]{"save/Const:0"}, new Tensor[]{cpPath}), new StringVector(), new StringVector("save/restore_all"), new TensorVector());
        checkStatus(status);
	}
	
	public void trainModel() {
	    TensorVector outputs = new TensorVector();
	    Status status = session.Run(new StringTensorPairVector(), new StringVector(), new StringVector("init"), outputs);
	    checkStatus(status);
	    
	    Tensor x = new Tensor(DT_FLOAT, new TensorShape(BATCH_SIZE, imgPixels));
        Tensor y_ = new Tensor(DT_FLOAT, new TensorShape(BATCH_SIZE, numClasses));

        float[] imgBatch = new float[BATCH_SIZE * imgPixels];
        float[] labelBatch = new float[BATCH_SIZE * numClasses];

        int iterations = testImages.length/BATCH_SIZE;
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
        Tensor y1_ = new Tensor(DT_FLOAT, new TensorShape(1, numClasses));

        float[] imgTest1 = new float[imgPixels];
        float[] labelTest1 = new float[numClasses];
        
        System.arraycopy(testImages, imgPixels * (n -1), imgTest1, 0, imgPixels);
        System.arraycopy(testLabels, numClasses * (n - 1), labelTest1, 0, numClasses);
        FloatBuffer xTest1 = x1.createBuffer();
        FloatBuffer yTest1 = y1_.createBuffer();
        
        xTest1.limit(imgPixels);
        yTest1.limit(numClasses);
        xTest1.put(imgTest1);
        yTest1.put(labelTest1);
        
        TensorVector outputs = new TensorVector();
        Status status = session.Run(new StringTensorPairVector(new String[] { "x", "y_" }, new Tensor[] { x1, y1_ }),
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
	
	private void checkStatus(Status status) {
	    if (!status.ok()) {
            throw new RuntimeException(status.error_message().getString());
        }
	}

}
