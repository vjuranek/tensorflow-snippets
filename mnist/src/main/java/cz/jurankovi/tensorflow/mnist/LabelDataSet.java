package cz.jurankovi.tensorflow.mnist;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

public class LabelDataSet extends DataSet {

	public static final int LABEL_MAGIC = 2049;

	private final byte[] labels;

	public LabelDataSet(String path) {
		super(path);
		labels = path.endsWith(".gz") ? loadDataSetGzip(path) : loadDataSet(path);
	}

	public byte[] getLabels() {
		return labels;
	}
	
	public int[][] asOneHot(int numClasses) {
		int[][] oneHot = new int[numClasses][labels.length];
		for (int i = 0; i < labels.length; i++) {
			oneHot[labels[i]][i] = 1;
		}
		return oneHot;
	}

	public void print(int n) {
		System.out.println(String.format("First %d labels: ", n));
		for (int i = 0; i < n; i++) {
			System.out.print(labels[i] + " ");
			if ((i + 1) % 10 == 0) {
				System.out.println();
			}
		}
	}

	protected byte[] loadDataSet(String path) {
		byte[] labels = new byte[0];
		try (FileInputStream fis = new FileInputStream(path)) {
			checkMagic(fis, LABEL_MAGIC);
			int numItems = readInt(fis);
			LOG.info(String.format("Trying to read %d items ...", numItems));
			labels = new byte[numItems];
			int read = fis.read(labels, 0, numItems);
			LOG.info(String.format("Labels read (read %d bytes)", read));
		} catch (IOException e) {
			LOG.severe(
					String.format("Something went wrong, cannot load data set '%s'. Cause: %s", path, e.getMessage()));
		}
		return labels;
	}

	protected byte[] loadDataSetGzip(String path) {
		byte[] labels = new byte[0];
		try (GZIPInputStream gzip = new GZIPInputStream(new FileInputStream(path), GZIP_BUFF_SIZE)) {
			checkMagic(gzip, LABEL_MAGIC);
			int numItems = readInt(gzip);
			LOG.info(String.format("Trying to read %d items ...", numItems));
			labels = new byte[numItems];
			byte[] buff = new byte[GZIP_BUFF_SIZE];
			int read = 0;
			int rb = 0;
			while ((rb = gzip.read(buff)) > -1) {
				System.arraycopy(buff, 0, labels, read, rb);
				read += rb;
			}
			LOG.info(String.format("Labels read (read %d bytes)", read));
		} catch (IOException e) {
			LOG.severe(
					String.format("Something went wrong, cannot load data set '%s'. Cause: %s", path, e.getMessage()));
		}
		return labels;
	}

}
