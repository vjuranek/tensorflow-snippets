package cz.jurankovi.tensorflow;

import static org.bytedeco.javacpp.tensorflow.DT_FLOAT;
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

public class MultiplyTest {

	public static void main(String args[]) throws Exception {
		InitMain("trainer", (int[]) null, null);

		GraphDef def = new GraphDef();
		ReadBinaryProto(Env.Default(), "/tmp/my-model/train.pb", def);
		SessionOptions options = new SessionOptions();
		try (final Session session = new Session(options)) {

			Status s = session.Create(def);
			if (!s.ok()) {
				throw new RuntimeException(s.error_message().getString());
			}

			Tensor a = new Tensor(DT_FLOAT, new TensorShape(3, 1));
			Tensor b = new Tensor(DT_FLOAT, new TensorShape(3, 1));
			FloatBuffer aVal = a.createBuffer();
			aVal.put(new float[] { 2.0f, 3.0f, 4.0f });
			FloatBuffer bVal = b.createBuffer();
			bVal.put(new float[] { 3.0f, 4.0f, 5.0f });

			Tensor c = new Tensor(DT_FLOAT, new TensorShape(3, 1));
			TensorVector outputs = new TensorVector();
			s = session.Run(
					new StringTensorPairVector(new String[] { "Placeholder", "Placeholder_1" }, new Tensor[] { a, b }),
					new StringVector("Mul"), new StringVector("Mul"), outputs);

			if (!s.ok()) {
				throw new RuntimeException(s.error_message().getString());
			}

			FloatBuffer output = outputs.get(0).createBuffer();
			for (int i = 0; i < output.limit(); i++) {
				System.out.println("output: " + output.get(i));
			}
		}

	}

}
