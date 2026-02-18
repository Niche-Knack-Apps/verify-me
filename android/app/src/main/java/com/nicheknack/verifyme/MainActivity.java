package com.nicheknack.verifyme;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(AudioRecorderPlugin.class);
        registerPlugin(FilePickerPlugin.class);
        registerPlugin(TTSEnginePlugin.class);
        registerPlugin(ModelManagerPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
