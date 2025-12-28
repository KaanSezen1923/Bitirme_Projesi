import requests


base_url="http://localhost:8000"

print("Süt Sihirbazı'na Hoşgeldiniz! Sorularınızı sorabilirsiniz. Çıkmak için 'exit', 'quit' veya 'çıkış' yazın.")
print("-----"*20)
while  True:
    try :
        user_input=input("User: ")
        if user_input.lower().strip() in ["exit","quit","çıkış"]:
            break
        print("Süt sihirbazı düşünüyor...........")
        result=requests.post(f"{base_url}/query",json={"question":user_input})
        result=result.json()
        
    except Exception as e:
        print("Bir hata oluştu:",str(e))
        continue

    classification=result["classification"]
    print(f"Sınıflandırma: {classification} \n")
    sql_query=result.get("sql_query","")
    if classification=="sql":
        print("Oluşturulan SQL Sorgusu:")
        print(sql_query,"\n")
        print("Sorgu Sonucu:")
        print(result["sql_result"],"\n")
    print("Süt Sihirbazı:",result["answer"])
    print("-----"*20)