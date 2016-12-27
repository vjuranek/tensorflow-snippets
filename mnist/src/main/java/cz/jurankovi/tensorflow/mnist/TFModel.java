package cz.jurankovi.tensorflow.mnist;

import static org.bytedeco.javacpp.tensorflow.DT_STRING;
import static org.bytedeco.javacpp.tensorflow.InitMain;
import static org.bytedeco.javacpp.tensorflow.ReadBinaryProto;

import org.bytedeco.javacpp.helper.tensorflow.StringArray;
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

public abstract class TFModel {

    public static final int BATCH_SIZE = 100;
    protected final String modelPath;
    protected final float[] trainLabels;
    protected final float[] trainImages;
    protected final float[] testLabels;
    protected final float[] testImages;
    protected final int imgPixels;
    protected final int numClasses;

    protected final Session session;

    public TFModel(String modelPath, int imgPixels, int numClasses, float[] trainLabesl, float[] trainImages,
            float[] testLabels, float[] testImages) {
        this.modelPath = modelPath;
        this.trainLabels = trainLabesl;
        this.trainImages = trainImages;
        this.testLabels = testLabels;
        this.testImages = testImages;
        this.imgPixels = imgPixels;
        this.numClasses = numClasses;

        InitMain("tf", (int[]) null, null);
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
        Status status = session.Run(
                new StringTensorPairVector(new String[] { "save/Const:0" }, new Tensor[] { cpPath }),
                new StringVector(), new StringVector("save/restore_all"), new TensorVector());
        checkStatus(status);
    }

    protected void checkStatus(Status status) {
        if (!status.ok()) {
            throw new RuntimeException(status.error_message().getString());
        }
    }

}
