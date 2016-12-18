package cz.jurankovi.tensorflow.mnist;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.logging.Logger;

public abstract class DataSet {

	public static final Logger LOG = Logger.getLogger(DataSet.class.getName());
	protected static final int GZIP_BUFF_SIZE = 8192;

	private final String path;

	public abstract void print(int n);

	public DataSet(String path) {
		this.path = path;
	}

	public String getPath() {
		return path;
	}

	protected void checkMagic(InputStream gzip, int expected) throws IOException {
		int magic = readInt(gzip);
		if (expected != magic) {
			throw new IllegalStateException(
					String.format("Data set magic number doesn't match! Expected %d, but got %d", expected, magic));
		}
	}

	protected int readInt(InputStream gzip) throws IOException {
		byte[] m = new byte[4];
		gzip.read(m);
		ByteBuffer bf = ByteBuffer.wrap(m);
		return bf.getInt();
	}

}
