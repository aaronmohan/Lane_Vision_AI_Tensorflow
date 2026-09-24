package com.aaron.lanevisionai;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.text.LineBreaker;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.Button;
import android.widget.ScrollView;
import android.widget.TextView;

public class InstructionsActivity extends AppCompatActivity {
    private static final int REQUEST_CODE_MAIN_ACTIVITY = 1;

    private Handler handler;
    private String text;
    private TextView textView;
    private ScrollView scrollView;

    Button skipButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_instructions);

        skipButton = (Button) findViewById(R.id.button_skip);

        skipButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent mainIntent = new Intent(InstructionsActivity.this, videoClass.class);
                if (isTaskRoot()) {
                    startActivityForResult(mainIntent, REQUEST_CODE_MAIN_ACTIVITY);
                } else {
                    finish();
                }
            }
        });

        textView = findViewById(R.id.textView4);
        textView.setText("");
        scrollView = findViewById(R.id.scrollView);
        text = (String) getText(R.string.instruction2);

        handler = new Handler();
        handler.postDelayed(new Runnable() {
            int index = 0;
            final CharSequence charSequence = text;
            final int length = charSequence.length();
            final int delay = 40;

            @Override
            public void run() {
                String builder = String.valueOf(textView.getText()) + charSequence.charAt(index);
                textView.setText(builder);

                scrollView.post(() -> scrollView.fullScroll(ScrollView.FOCUS_DOWN));

                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                    textView.setJustificationMode(LineBreaker.JUSTIFICATION_MODE_INTER_WORD);
                }

                if (index < length - 1) {
                    index++;
                    handler.postDelayed(this, delay);
                }
            }
        }, 2000);
    }
}
