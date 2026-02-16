package com.nicheknack.verifyme;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import androidx.activity.result.ActivityResult;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.OutputStream;

@CapacitorPlugin(name = "FilePicker")
public class FilePickerPlugin extends Plugin {

    private byte[] pendingAudioData = null;

    @PluginMethod
    public void saveAudioFile(PluginCall call) {
        String filename = call.getString("filename", "output.wav");
        String mimeType = call.getString("mimeType", "audio/wav");
        String sourcePath = call.getString("sourcePath", "");

        try {
            java.io.File sourceFile = new java.io.File(sourcePath);
            if (sourceFile.exists()) {
                pendingAudioData = java.nio.file.Files.readAllBytes(sourceFile.toPath());
            }
        } catch (Exception e) {
            JSObject ret = new JSObject();
            ret.put("success", false);
            ret.put("error", "Could not read source file: " + e.getMessage());
            call.resolve(ret);
            return;
        }

        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType(mimeType);
        intent.putExtra(Intent.EXTRA_TITLE, filename);

        startActivityForResult(call, intent, "handleSaveResult");
    }

    @ActivityCallback
    private void handleSaveResult(PluginCall call, ActivityResult result) {
        if (call == null) return;

        if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
            Uri uri = result.getData().getData();

            if (uri != null && pendingAudioData != null) {
                try {
                    OutputStream outputStream = getContext().getContentResolver().openOutputStream(uri);
                    if (outputStream != null) {
                        outputStream.write(pendingAudioData);
                        outputStream.close();

                        JSObject ret = new JSObject();
                        ret.put("success", true);
                        ret.put("uri", uri.toString());
                        call.resolve(ret);
                    } else {
                        JSObject ret = new JSObject();
                        ret.put("success", false);
                        ret.put("error", "Could not open output stream");
                        call.resolve(ret);
                    }
                } catch (Exception e) {
                    JSObject ret = new JSObject();
                    ret.put("success", false);
                    ret.put("error", "Error writing file: " + e.getMessage());
                    call.resolve(ret);
                }
            } else {
                JSObject ret = new JSObject();
                ret.put("success", false);
                ret.put("error", "No URI or audio data");
                call.resolve(ret);
            }
        } else {
            JSObject ret = new JSObject();
            ret.put("success", false);
            ret.put("error", "cancelled");
            call.resolve(ret);
        }

        pendingAudioData = null;
    }

    @PluginMethod
    public void pickDirectory(PluginCall call) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        startActivityForResult(call, intent, "handleDirectoryResult");
    }

    @ActivityCallback
    private void handleDirectoryResult(PluginCall call, ActivityResult result) {
        if (call == null) return;

        if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
            Uri uri = result.getData().getData();
            if (uri != null) {
                // Take persistable permission
                getContext().getContentResolver().takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                );

                JSObject ret = new JSObject();
                ret.put("success", true);
                ret.put("uri", uri.toString());
                call.resolve(ret);
            } else {
                JSObject ret = new JSObject();
                ret.put("success", false);
                ret.put("error", "No URI received");
                call.resolve(ret);
            }
        } else {
            JSObject ret = new JSObject();
            ret.put("success", false);
            ret.put("cancelled", true);
            call.resolve(ret);
        }
    }
}
