package cz.jurankovi.tensorflow.mnist;

import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.DataBufferInt;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class Util {

	public static final int[] BAND_MASKS = { 0xFF0000, 0xFF00, 0xFF, 0xFF000000 };

	public static void saveAsJPG(int width, int height, int[] pixels, String path) throws IOException {
		DataBufferInt buffer = new DataBufferInt(pixels, pixels.length);
		WritableRaster raster = Raster.createPackedRaster(buffer, width, height, width, BAND_MASKS, null);
		ColorModel cm = ColorModel.getRGBdefault();
		BufferedImage image = new BufferedImage(cm, raster, cm.isAlphaPremultiplied(), null);
		ImageIO.write(image, "JPG", new File(path));
	}

}
