package cz.jurankovi.tensorflow;

import static org.bytedeco.javacpp.tensorflow.DT_DOUBLE;
import static org.bytedeco.javacpp.tensorflow.InitMain;
import static org.bytedeco.javacpp.tensorflow.ReadBinaryProto;

import java.nio.DoubleBuffer;

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

public class MultiplyTest {

	public static void main(String args[]) throws Exception {
		InitMain("trainer", (int[]) null, null);

		GraphDef def = new GraphDef();
		ReadBinaryProto(Env.Default(), "/tmp/my-model/basic.pb", def);
		SessionOptions options = new SessionOptions();
		try (final Session session = new Session(options)) {
			Status sess  = session.Create(def);
			
			Tensor x = new Tensor(DT_DOUBLE, new TensorShape(3, 1));
			Tensor y = new Tensor(DT_DOUBLE, new TensorShape(3, 1));
			DoubleBuffer xBuff = x.createBuffer();
			xBuff.put(new double[] { 2.0, 3.0, 4.0 });
			DoubleBuffer yBuff = y.createBuffer();
			yBuff.put(new double[] { 3.0, 4.0, 5.0 });

			TensorVector outputs = new TensorVector();
			sess = session.Run(
					new StringTensorPairVector(new String[] { "Placeholder", "Placeholder_1" }, new Tensor[] { x, y }),
					new StringVector("Mul"), new StringVector("Mul"), outputs);

			if (!sess.ok()) {
				throw new RuntimeException(sess.error_message().getString());
			}

			DoubleBuffer output = outputs.get(0).createBuffer();
			for (int i = 0; i < output.limit(); i++) {
				System.out.println("output: " + output.get(i));
			}
		}

	}

}
