#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integração entre Modelo de Frete e LM Studio
Permite usar o modelo ML para cálculos e gerar prompts para LM Studio
"""

import pickle
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional

class FreteLMIntegration:
    """
    Classe para integrar modelo de frete com LM Studio
    """
    
    def __init__(self, model_path: str = "models/frete_predictor.pkl"):
        """Inicializar com modelo treinado"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            
            print("✅ Modelo de frete carregado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def predict_frete(self, volume_ton: float, distancia_km: float, mes: int, 
                     trimestre: int, ano: int, modal: str, tipo_rodovia: str, 
                     tipo_veiculo: str) -> float:
        """
        Fazer previsão de frete
        
        Args:
            volume_ton: Volume em toneladas
            distancia_km: Distância em quilômetros
            mes: Mês (1-12)
            trimestre: Trimestre (1-4)
            ano: Ano
            modal: Modal de transporte
            tipo_rodovia: Tipo de rodovia
            tipo_veiculo: Tipo de veículo
        
        Returns:
            Preço por ton/km previsto
        """
        try:
            # Features numéricas
            X_numeric = np.array([[volume_ton, distancia_km, mes, trimestre, ano]])
            
            # Codificar features categóricas
            modal_encoded = self.label_encoders['modal'].transform([modal])[0]
            rodovia_encoded = self.label_encoders['tipo_rodovia'].transform([tipo_rodovia])[0]
            veiculo_encoded = self.label_encoders['tipo_veiculo'].transform([tipo_veiculo])[0]
            
            # Combinar features
            X = np.hstack([X_numeric, np.array([[modal_encoded, rodovia_encoded, veiculo_encoded]])])
            
            # Normalizar
            X_scaled = self.scaler.transform(X)
            
            # Prever
            prediction = self.model.predict(X_scaled)[0]
            return prediction
            
        except Exception as e:
            print(f"❌ Erro na previsão: {e}")
            return None
    
    def get_frete_insights(self, volume_ton: float, distancia_km: float, 
                          modal: str, tipo_rodovia: str, tipo_veiculo: str) -> Dict[str, Any]:
        """
        Obter insights completos sobre o frete
        
        Args:
            volume_ton: Volume em toneladas
            distancia_km: Distância em quilômetros
            modal: Modal de transporte
            tipo_rodovia: Tipo de rodovia
            tipo_veiculo: Tipo de veículo
        
        Returns:
            Dicionário com insights, preços e recomendações
        """
        # Fazer previsão
        prediction = self.predict_frete(volume_ton, distancia_km, 6, 2, 2024, 
                                      modal, tipo_rodovia, tipo_veiculo)
        
        if prediction is None:
            return None
        
        # Calcular preço total
        preco_total = prediction * volume_ton * distancia_km
        
        # Gerar insights
        insights = {
            'preco_ton_km': prediction,
            'preco_total': preco_total,
            'volume_ton': volume_ton,
            'distancia_km': distancia_km,
            'modal': modal,
            'tipo_rodovia': tipo_rodovia,
            'tipo_veiculo': tipo_veiculo,
            'recomendacoes': [],
            'alertas': [],
            'oportunidades': []
        }
        
        # Análise de volume
        if volume_ton > 100:
            insights['recomendacoes'].append("Volume alto - considere modal ferroviário para economia")
            insights['oportunidades'].append("Negociar contratos de longo prazo com transportadoras")
        elif volume_ton < 25:
            insights['recomendacoes'].append("Volume baixo - modal rodoviário é mais adequado")
            insights['alertas'].append("Custo por ton pode ser alto devido ao volume baixo")
        
        # Análise de distância
        if distancia_km > 1000:
            insights['recomendacoes'].append("Distância longa - avalie modal ferroviário ou aquaviário")
            insights['oportunidades'].append("Consolidar cargas para otimizar custos")
        elif distancia_km < 100:
            insights['recomendacoes'].append("Distância curta - modal rodoviário é ideal")
        
        # Análise de modal
        if modal == "RODOVIARIO" and distancia_km > 800:
            insights['alertas'].append("Distância alta para modal rodoviário - considere alternativas")
        elif modal == "FERROVIARIO" and distancia_km < 300:
            insights['alertas'].append("Distância baixa para modal ferroviário - pode não ser viável")
        
        # Análise de rodovia
        if tipo_rodovia == "PEDAGIO" and distancia_km > 500:
            insights['recomendacoes'].append("Avaliar rotas alternativas para reduzir custos de pedágio")
        
        # Análise de veículo
        if tipo_veiculo == "TRUCK" and volume_ton > 80:
            insights['recomendacoes'].append("Volume alto para truck - considere carreta ou múltiplos veículos")
        
        return insights
    
    def generate_lm_prompt(self, user_question: str, insights: Dict[str, Any]) -> str:
        """
        Gerar prompt otimizado para LM Studio
        
        Args:
            user_question: Pergunta do usuário
            insights: Insights do modelo de frete
        
        Returns:
            Prompt formatado para LM Studio
        """
        prompt = f"""
ANÁLISE DE FRETE - RESULTADO DO MODELO ML

PERGUNTA DO USUÁRIO: {user_question}

DADOS DO MODELO ML:
- Preço por ton/km: R$ {insights['preco_ton_km']:.4f}
- Preço total: R$ {insights['preco_total']:.2f}
- Volume: {insights['volume_ton']} toneladas
- Distância: {insights['distancia_km']} km
- Modal: {insights['modal']}
- Tipo de rodovia: {insights['tipo_rodovia']}
- Tipo de veículo: {insights['tipo_veiculo']}

ANÁLISE AUTOMÁTICA:
- Recomendações: {', '.join(insights['recomendacoes']) if insights['recomendacoes'] else 'Nenhuma recomendação específica'}
- Alertas: {', '.join(insights['alertas']) if insights['alertas'] else 'Nenhum alerta'}
- Oportunidades: {', '.join(insights['oportunidades']) if insights['oportunidades'] else 'Nenhuma oportunidade identificada'}

INSTRUÇÕES PARA LM STUDIO:
Baseado nos dados acima, forneça:
1. Análise detalhada da situação
2. Recomendações práticas e acionáveis
3. Comparação com alternativas
4. Estratégias de otimização
5. Próximos passos recomendados

Responda de forma técnica mas acessível, sempre referenciando os dados do modelo ML.
"""
        return prompt
    
    def create_frete_report(self, insights: Dict[str, Any]) -> str:
        """
        Criar relatório completo de frete
        
        Args:
            insights: Insights do modelo
        
        Returns:
            Relatório formatado
        """
        report = f"""
🚛 RELATÓRIO DE ANÁLISE DE FRETE
{'='*50}

📊 DADOS BÁSICOS:
   • Volume: {insights['volume_ton']} toneladas
   • Distância: {insights['distancia_km']} km
   • Modal: {insights['modal']}
   • Rodovia: {insights['tipo_rodovia']}
   • Veículo: {insights['tipo_veiculo']}

💰 ANÁLISE DE CUSTOS:
   • Preço por ton/km: R$ {insights['preco_ton_km']:.4f}
   • Preço total do frete: R$ {insights['preco_total']:.2f}

🎯 RECOMENDAÇÕES:
"""
        
        if insights['recomendacoes']:
            for i, rec in enumerate(insights['recomendacoes'], 1):
                report += f"   {i}. {rec}\n"
        else:
            report += "   • Nenhuma recomendação específica\n"
        
        report += f"""
⚠️ ALERTAS:
"""
        
        if insights['alertas']:
            for i, alerta in enumerate(insights['alertas'], 1):
                report += f"   {i}. {alerta}\n"
        else:
            report += "   • Nenhum alerta identificado\n"
        
        report += f"""
💡 OPORTUNIDADES:
"""
        
        if insights['oportunidades']:
            for i, op in enumerate(insights['oportunidades'], 1):
                report += f"   {i}. {op}\n"
        else:
            report += "   • Nenhuma oportunidade específica identificada\n"
        
        report += f"""
📈 PRÓXIMOS PASSOS:
1. Validar previsão com dados históricos
2. Comparar com cotações de mercado
3. Implementar recomendações prioritárias
4. Monitorar resultados e ajustar estratégia

{'='*50}
Relatório gerado automaticamente pelo modelo ML
"""
        
        return report

def main():
    """Função principal para demonstração"""
    try:
        # Inicializar integração
        integration = FreteLMIntegration()
        
        print("🎯 DEMONSTRAÇÃO DA INTEGRAÇÃO FRETE + LM STUDIO")
        print("="*60)
        
        # Mostrar valores disponíveis nos encoders
        print("\n📋 VALORES DISPONÍVEIS NOS ENCODERS:")
        print(f"   Modal: {integration.label_encoders['modal'].classes_}")
        print(f"   Rodovia: {integration.label_encoders['tipo_rodovia'].classes_}")
        print(f"   Veículo: {integration.label_encoders['tipo_veiculo'].classes_}")
        
        # Exemplo 1: Análise básica com valores corretos
        print("\n📊 EXEMPLO 1: Análise de frete rodoviário")
        insights1 = integration.get_frete_insights(
            volume_ton=75,
            distancia_km=600,
            modal="Spot",
            tipo_rodovia="Rodovia",
            tipo_veiculo="Truck"
        )
        
        if insights1:
            print(f"   Preço por ton/km: R$ {insights1['preco_ton_km']:.4f}")
            print(f"   Preço total: R$ {insights1['preco_total']:.2f}")
            print(f"   Recomendações: {len(insights1['recomendacoes'])}")
        
        # Exemplo 2: Análise com carreta
        print("\n📊 EXEMPLO 2: Análise de frete com carreta")
        insights2 = integration.get_frete_insights(
            volume_ton=150,
            distancia_km=1200,
            modal="Spot",
            tipo_rodovia="Rodovia",
            tipo_veiculo="Carreta"
        )
        
        if insights2:
            print(f"   Preço por ton/km: R$ {insights2['preco_ton_km']:.4f}")
            print(f"   Preço total: R$ {insights2['preco_total']:.2f}")
            print(f"   Recomendações: {len(insights2['recomendacoes'])}")
        
        # Exemplo 3: Gerar prompt para LM Studio
        print("\n📝 EXEMPLO 3: Prompt para LM Studio")
        if insights1:
            prompt = integration.generate_lm_prompt(
                "Como otimizar este frete de 75 ton por 600 km?",
                insights1
            )
            print("   Prompt gerado com sucesso!")
            print(f"   Tamanho: {len(prompt)} caracteres")
        
        # Exemplo 4: Relatório completo
        print("\n📋 EXEMPLO 4: Relatório completo")
        if insights1:
            report = integration.create_frete_report(insights1)
            print("   Relatório gerado com sucesso!")
            print(f"   Tamanho: {len(report)} caracteres")
        
        print("\n✅ Demonstração concluída com sucesso!")
        print("\n🚀 PRÓXIMOS PASSOS:")
        print("1. Use get_frete_insights() para análises")
        print("2. Use generate_lm_prompt() para LM Studio")
        print("3. Use create_frete_report() para relatórios")
        print("4. Integre com sua aplicação preferida")
        
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")

if __name__ == "__main__":
    main()
