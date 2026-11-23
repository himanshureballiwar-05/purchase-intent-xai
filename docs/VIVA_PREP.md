# Viva / Mentor Q&A Prep (Hinglish)

Is file ko padh ke apne words me bolna seekho. Mentor "kyun" puchega - isliye har jawab me reason hai.

## 30-second pitch
"Maine e-commerce sessions ka data lekar ek model banaya jo predict karta hai ki visitor kharidega ya nahi, aur SHAP/LIME se
batata hai kyun. 6 algorithms compare kiye, imbalance 3 tareeke se handle kiya, best model CatBoost nikla (PR-AUC 0.86).
Isko FastAPI API aur Streamlit dashboard se serve kiya, Docker, tests aur CI ke saath."

## Sawal-jawab

**1. Accuracy kyun nahi use ki?**
Sirf 15.7% log kharidte hain. "Koi nahi kharidega" bolne wala model bhi 84.3% accurate hota hai par ek bhi buyer nahi pakadta.
Isliye PR-AUC aur recall dekhe.

**2. PR-AUC kya hota hai?**
Precision-Recall curve ke neeche ka area. Imbalanced data me ROC-AUC kabhi-kabhi zyada achha dikhata hai; PR-AUC rare class (buyers)
par focus karta hai. Ye threshold pe depend nahi karta, isliye models ko fairly compare kar sakte hain.

**3. SMOTE kya hai, aur tumne kahan lagaya?**
Minority class ke naye synthetic samples banata hai. Maine usse **Pipeline ke andar** lagaya, taaki cross-validation me sirf training
folds pe chale. Poore data pe pehle lagate to validation ka info leak hota aur score jhootha badh jata.
(Result: is data me SMOTE ne PR-AUC thoda kam hi kiya.)

**4. Data leakage kya hai? Tumne kaise roka?**
Jab model ko training me aisi info mil jaye jo real time pe nahi hogi. Maine: split pehle kiya, encoding/scaling/SMOTE pipeline me rakhe,
test set sirf ek baar use kiya, threshold train ke out-of-fold predictions se chuna.

**5. PageValues me kya problem hai?** (sabse important)
Ye feature sabse strong hai, par checkout ke paas ka signal hai: jitne buyers hain sab ke sab me PageValues > 0 hai. Isko hatane par
PR-AUC 0.85 se 0.53 pe gir jata hai. Maine config me switch rakha, ablation chalaya, aur model card me likha ki model late-session
scoring ke liye best hai.

**6. Integer columns (Browser, Region...) ko scale kyun nahi kiya?**
Wo categories ke codes hain, quantity nahi. Browser 4 "Browser 2 ka double" nahi hota. Isliye one-hot encode kiya
(`handle_unknown="ignore"` taaki naya code aaye to crash na ho).

**7. Threshold 0.5 kyun nahi?**
Nudge ki cost aur saved purchase ki value alag hoti hai. Break-even probability = cost / (value x success rate) = 3 / (60 x 0.25) = 0.20.
Maine out-of-fold predictions pe expected net value maximize karke 0.14 chuna (curve 0.14-0.20 ke beech almost flat hai).
Ye numbers assumptions hain, real nahi - model card me bhi likha hai.

**8. SHAP aur LIME me kya fark hai?**
SHAP har feature ka exact additive contribution deta hai (tree models ke liye exact); poore data pe average karke global importance
milti hai. LIME ek prediction ke aas-paas chhota simple model fit karta hai - approximate hai. Humne dono dikhaye; PageValues pe dono
agree karte hain, baaki me SHAP pe zyada bharosa.

**9. Explainability ka matlab causation hai?**
Nahi. Ye batata hai model kaise sochta hai, ye nahi ki feature badalne se purchase "cause" hoga.

**10. CatBoost kyun jeeta? Baaki kitne door the?**
CV PR-AUC: CatBoost ~0.844, XGBoost ~0.843, LightGBM ~0.836, Logistic Regression ~0.831 - farak chhota hai. Tuning ke baad CatBoost 0.850.
Honestly top models almost barabar hain; logistic regression bhi strong baseline hai.

**11. Cross-validation kya hai? Kyun 5-fold?**
Training data ko 5 hisso me baant ke har baar 1 hissa validation banaya. Stratified isliye ki har fold me 15.7% buyers rahein.

**12. Train/test split kaise kiya?**
80/20, stratified, seed 42. Test set pe model choose nahi kiya, sirf final report ke liye use kiya.

**13. API me validation kaise hoti hai?**
Pydantic schema: range check (BounceRates 0-1), Month/VisitorType allowed values. Galat input pe 422. Model load nahi hua to 503.

**14. Model API me ek baar kyun load hota hai?**
Har request pe load karna slow hota; startup pe ek baar load karte hain (lifespan).

**15. Docker kyun?**
"Mere laptop pe chalta hai" wali problem khatam - same environment kahin bhi. `docker compose up` se API + dashboard dono chalte hain.

**16. MLflow kya karta hai?**
Har experiment ke params, metrics aur model ka record rakhta hai, taaki compare aur reproduce kar sakein. Final model registry me register hota hai.

**17. CI/CD kya hai?**
GitHub Actions: har push pe lint, format check, tests aur Docker build automatically chalte hain.

**18. Limitations?**
Ek dataset snapshot (seasonality), PageValues dependence, business numbers assumptions, aur ye model individuals ke baare me
sensitive decisions ke liye nahi - sirf marketing nudge jaise low-risk decisions ke liye.

**19. Agar sab features hata ke simple banana ho to?**
Logistic Regression + class weights bhi ~0.83 PR-AUC deta hai - simple, tez, interpretable.

**20. Aage kya improve karoge?**
Probability calibration, early-scoring model (PageValues ke bina) ke liye better features, monitoring/drift detection, A/B test se
intervention ka real success rate measure karna.

## Tum khud karo (zaruri)
- Code ko ek baar `src/models/train.py` se `src/api/main.py` tak padho.
- Dashboard chalake 3-4 predictions try karo, PageValues badal ke dekho kya hota hai.
- Notebooks run karke outputs apne words me samjho.
- GitHub pe apne naam se push karo, chhote commits ke saath.
