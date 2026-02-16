package com.nicheknack.verifyme;

import android.Manifest;
import android.media.MediaRecorder;
import android.os.Environment;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import java.io.File;
import java.io.IOException;

@CapacitorPlugin(
    name = "AudioRecorder",
    permissions = {
        @Permission(strings = {Manifest.permission.RECORD_AUDIO}, alias = "microphone")
    }
)
public class AudioRecorderPlugin extends Plugin {

    private MediaRecorder recorder = null;
    private String currentFilePath = null;

    @PluginMethod
    public void startRecording(PluginCall call) {
        if (!hasPermission(Manifest.permission.RECORD_AUDIO)) {
            requestPermissionForAlias("microphone", call, "handleMicPermission");
            return;
        }

        doStartRecording(call);
    }

    @com.getcapacitor.annotation.ActivityCallback
    private void handleMicPermission(PluginCall call) {
        if (hasPermission(Manifest.permission.RECORD_AUDIO)) {
            doStartRecording(call);
        } else {
            JSObject ret = new JSObject();
            ret.put("success", false);
            ret.put("error", "Microphone permission denied");
            call.resolve(ret);
        }
    }

    private void doStartRecording(PluginCall call) {
        try {
            File outputDir = getContext().getCacheDir();
            File outputFile = File.createTempFile("voice_sample_", ".m4a", outputDir);
            currentFilePath = outputFile.getAbsolutePath();

            recorder = new MediaRecorder();
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            recorder.setAudioSamplingRate(44100);
            recorder.setAudioEncodingBitRate(128000);
            recorder.setOutputFile(currentFilePath);
            recorder.prepare();
            recorder.start();

            JSObject ret = new JSObject();
            ret.put("success", true);
            ret.put("recording", true);
            call.resolve(ret);
        } catch (IOException e) {
            JSObject ret = new JSObject();
            ret.put("success", false);
            ret.put("error", "Failed to start recording: " + e.getMessage());
            call.resolve(ret);
        }
    }

    @PluginMethod
    public void stopRecording(PluginCall call) {
        if (recorder == null) {
            JSObject ret = new JSObject();
            ret.put("success", false);
            ret.put("error", "No active recording");
            call.resolve(ret);
            return;
        }

        try {
            recorder.stop();
            recorder.release();
            recorder = null;

            JSObject ret = new JSObject();
            ret.put("success", true);
            ret.put("filePath", currentFilePath);
            call.resolve(ret);
        } catch (Exception e) {
            recorder = null;
            JSObject ret = new JSObject();
            ret.put("success", false);
            ret.put("error", "Failed to stop recording: " + e.getMessage());
            call.resolve(ret);
        }
    }

    @PluginMethod
    public void pickAudioFile(PluginCall call) {
        android.content.Intent intent = new android.content.Intent(android.content.Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(android.content.Intent.CATEGORY_OPENABLE);
        intent.setType("audio/*");

        startActivityForResult(call, intent, "handleAudioResult");
    }

    @com.getcapacitor.annotation.ActivityCallback
    private void handleAudioResult(PluginCall call, androidx.activity.result.ActivityResult result) {
        if (call == null) return;

        if (result.getResultCode() == android.app.Activity.RESULT_OK && result.getData() != null) {
            android.net.Uri uri = result.getData().getData();
            if (uri != null) {
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
