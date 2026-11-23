$tests = @(
    @{ label = "HUMAN"; text = "I went to the market yesterday and honestly the tomatoes looked terrible. My neighbor said the same thing happened last week." },
    @{ label = "HUMAN"; text = "ugh my laptop died right before the presentation so I had to borrow Priya's and it kept lagging lol." },
    @{ label = "HUMAN"; text = "Grandma's kitchen always smelled like cardamom and burnt sugar. She never measured anything, just tossed things in a pot." },
    @{ label = "HUMAN"; text = "Went to the new bakery on 5th street this morning. The croissants were decent but way too buttery for my taste." },
    @{ label = "AI";    text = "Artificial intelligence is transforming many industries by automating routine tasks, improving decision-making, and enabling new forms of creativity." },
    @{ label = "AI";    text = "Remote work has reshaped the modern workplace in several important ways. It offers flexibility and reduces commuting time, but it can also blur boundaries." },
    @{ label = "AI";    text = "Climate change poses a complex challenge that requires coordinated action across governments, businesses, and individuals." },
    @{ label = "AI";    text = "Effective leadership depends on clear communication, emotional intelligence, and a willingness to adapt to changing circumstances." }
)

foreach ($t in $tests) {
    $body = @{ text = $t.text } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://localhost:8000/api/text/analyze" -Method Post -ContentType "application/json" -Body $body
    $score = [math]::Round($r.desklib.ai_confidence, 3)
    Write-Output ("{0,-6} score={1,-6} {2}" -f $t.label, $score, $t.text.Substring(0, 50))
}