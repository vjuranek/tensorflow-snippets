package cz.jurankovi.tensorflow.mnist;

public class Mnist {

	public static void main(String[] args) throws Exception {
		LabelDataSet labels = new LabelDataSet("/tmp/mnist/data/train-labels-idx1-ubyte.gz");
		ImageDataSet images = new ImageDataSet("/tmp/mnist/data/train-images-idx3-ubyte.gz");
		labels.print(100);
		images.print(5);

		int[][] imgs = images.getRawImages().get2d();
		for (int i = 0; i < 100; i++) {
			Util.saveAsJPG(28, 28, imgs[i], String.format("/tmp/mnist/jpg/%03d.jpg", i));
		}
	}

}
